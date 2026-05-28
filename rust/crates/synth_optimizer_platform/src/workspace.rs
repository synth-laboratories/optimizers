use std::fs;
use std::path::{Path, PathBuf};

use std::collections::BTreeMap;

use rusqlite::{params, Connection, OptionalExtension};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use time::OffsetDateTime;

use crate::artifacts::{ArtifactPaths, ArtifactRef};
use crate::cache::{stable_json, CacheAccessRecord, CacheMode, CacheProfileRecord};
use crate::candidates::{
    AcceptanceDecisionInput, AcceptanceDecisionRecord, CandidateDeltaInput, CandidateDeltaRecord,
    CandidatePayloadInput, CandidatePayloadRecord, FrontierCellInput, FrontierCellRecord,
    PlanLinkInput, PlanLinkRecord,
};
use crate::checkpoints::CheckpointRecord;
use crate::config::SynthOptimizerConfig;
use crate::data_models::{
    materialization_record_json, record_json, EvaluationCacheRecord, MaterializationRecord,
};
use crate::error::{OptimizerError, Result};
use crate::events::EventStreamRecord;
use crate::evidence::{
    EvidenceFrame, SensorDerivedRecords, SubagentInvocation, TraceAnnotation, VerifierJob,
};
use crate::invariants::{
    CountMismatchInput, InvariantReport, InvariantViolation, InvariantViolationInput,
};
use crate::jobs::{OptimizerJob, OptimizerJobKind, OptimizerJobStatus, RetryPolicy};
use crate::limits::{
    BudgetCommitRecord, BudgetLedgerSnapshot, BudgetLedgerTotals, BudgetReleaseRecord,
    BudgetReservationRecord, RunLimitsRecord, RuntimeEffectAdmissionRecord,
};
use crate::operations::OperationRecord;
use crate::projections::ProjectionFreshnessRecord;
use crate::resources::{ResourceLeaseRecord, ResourceLeaseRecordInput};
use crate::rollouts::{RolloutEventRecord, RolloutRecord, SensorRolloutRecords};
use crate::runtime_records::{
    runtime_record_json, ContainerContractSnapshotRecord, DatasetSnapshotRecord,
    PromptProgramSnapshotRecord, RenderedOptimizerStateInput, RenderedOptimizerStateRecord,
    ResolvedRunConfigRecord, RuntimeEffectRecord,
};
use crate::scores::{
    ObjectiveSetRecord, ObjectiveSpec, ParetoComparisonRecord, ScoreRecord, ScoreVectorRecord,
    SensorScoreRecords,
};
use crate::sensors::SensorFrame;
use crate::state_machine::OptimizerTransition;
use crate::stopper::StopperStateRecord;
use crate::usage::UsageLedgerRecord;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkspaceStatus {
    pub schema_version: String,
    pub path: String,
    pub run_requests: Vec<WorkspaceRunRequestStatus>,
    pub run_request_status_counts: BTreeMap<String, u64>,
    pub operation_status_counts: BTreeMap<String, u64>,
    pub resource_lease_status_counts: BTreeMap<String, u64>,
    pub resource_lease_kind_counts: BTreeMap<String, u64>,
    pub projection_status_counts: BTreeMap<String, u64>,
    pub invariant_status_counts: BTreeMap<String, u64>,
    pub invariant_violation_severity_counts: BTreeMap<String, u64>,
    pub workspace_projection_freshness: Vec<ProjectionFreshnessRecord>,
    pub workspace_invariant_report: Option<InvariantReport>,
    pub runs: Vec<WorkspaceRunStatus>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkspaceRunRequestStatus {
    pub request_id: String,
    pub run_id: String,
    pub status: String,
    pub config_path: String,
    pub container_url: String,
    pub cache_mode: String,
    pub cache_namespace: String,
    pub output_dir: String,
    pub run_dir: String,
    pub priority: i64,
    pub submitted_at: String,
    pub leased_at: Option<String>,
    pub lease_expires_at: Option<String>,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
    pub updated_at: String,
    pub lease_id: Option<String>,
    pub worker_id: Option<String>,
    pub run_workspace_db_path: Option<String>,
    pub result_manifest_path: Option<String>,
    pub best_candidate_id: Option<String>,
    pub cost_usd: Option<f64>,
    pub usage: Value,
    pub result: Value,
    pub error: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkspaceRunStatus {
    pub run_id: String,
    pub state: String,
    pub cache_mode: String,
    pub cache_namespace: String,
    pub output_dir: String,
    pub run_dir: String,
    pub manifest_path: String,
    pub best_candidate_id: Option<String>,
    pub cost_usd: Option<f64>,
    pub usage: Value,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub updated_at: String,
    pub counts: WorkspaceEntityCounts,
    pub candidate_status_counts: BTreeMap<String, u64>,
    pub candidate_delta_operation_counts: BTreeMap<String, u64>,
    pub acceptance_decision_counts: BTreeMap<String, u64>,
    pub plan_link_relation_counts: BTreeMap<String, u64>,
    pub frontier_cell_status_counts: BTreeMap<String, u64>,
    pub cache_access_action_counts: BTreeMap<String, u64>,
    pub cache_access_boundary_counts: BTreeMap<String, u64>,
    pub event_stream_event_type_counts: BTreeMap<String, u64>,
    pub usage_ledger_boundary_counts: BTreeMap<String, u64>,
    pub stopper_state_status_counts: BTreeMap<String, u64>,
    pub checkpoint_status_counts: BTreeMap<String, u64>,
    pub checkpoint_kind_counts: BTreeMap<String, u64>,
    pub optimizer_job_status_counts: BTreeMap<String, u64>,
    pub optimizer_job_kind_counts: BTreeMap<String, u64>,
    pub optimizer_job_lease_counts: BTreeMap<String, u64>,
    pub rollout_job_status_counts: BTreeMap<String, u64>,
    pub rollout_stage_counts: BTreeMap<String, u64>,
    pub rollout_status_counts: BTreeMap<String, u64>,
    pub rollout_record_stage_counts: BTreeMap<String, u64>,
    pub rollout_event_type_counts: BTreeMap<String, u64>,
    pub operation_status_counts: BTreeMap<String, u64>,
    pub resource_lease_status_counts: BTreeMap<String, u64>,
    pub resource_lease_kind_counts: BTreeMap<String, u64>,
    pub score_objective_counts: BTreeMap<String, u64>,
    pub score_stage_counts: BTreeMap<String, u64>,
    pub trace_annotation_status_counts: BTreeMap<String, u64>,
    pub evidence_frame_kind_counts: BTreeMap<String, u64>,
    pub verifier_job_status_counts: BTreeMap<String, u64>,
    pub subagent_status_counts: BTreeMap<String, u64>,
    pub projection_freshness: Vec<ProjectionFreshnessRecord>,
    pub invariant_report: Option<InvariantReport>,
    pub invariant_violation_severity_counts: BTreeMap<String, u64>,
    pub latest_transition: Option<WorkspaceStateTransitionStatus>,
    pub latest_rendered_state: Option<RenderedOptimizerStateRecord>,
    pub rendered_state_phase_counts: BTreeMap<String, u64>,
    pub budget_status: Value,
    pub manifest_present: bool,
    pub runtime_effect_status_counts: BTreeMap<String, u64>,
    pub runtime_effect_lane_counts: BTreeMap<String, u64>,
    pub runtime_effect_failure_class_counts: BTreeMap<String, u64>,
    pub runtime_effect_admission_status_counts: BTreeMap<String, u64>,
    pub budget_reservation_status_counts: BTreeMap<String, u64>,
    pub budget_releases: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WorkspaceEntityCounts {
    pub resolved_run_configs: u64,
    pub container_contract_snapshots: u64,
    pub prompt_program_snapshots: u64,
    pub dataset_snapshots: u64,
    pub run_limits: u64,
    pub rendered_optimizer_states: u64,
    pub runtime_effects: u64,
    pub runtime_effect_admissions: u64,
    pub budget_reservations: u64,
    pub budget_commits: u64,
    pub budget_releases: u64,
    pub candidates: u64,
    pub candidate_payloads: u64,
    pub candidate_deltas: u64,
    pub acceptance_decisions: u64,
    pub frontier_cells: u64,
    pub plan_links: u64,
    pub cache_profiles: u64,
    pub cache_accesses: u64,
    pub materializations: u64,
    pub evaluation_cache: u64,
    pub event_stream_events: u64,
    pub usage_ledger: u64,
    pub stopper_states: u64,
    pub checkpoints: u64,
    pub optimizer_jobs: u64,
    pub rollout_jobs: u64,
    pub rollouts: u64,
    pub rollout_events: u64,
    pub operations: u64,
    pub resource_leases: u64,
    pub objective_sets: u64,
    pub objectives: u64,
    pub scores: u64,
    pub score_vectors: u64,
    pub pareto_comparisons: u64,
    pub sensor_frames: u64,
    pub trace_annotations: u64,
    pub evidence_frames: u64,
    pub verifier_jobs: u64,
    pub subagent_invocations: u64,
    pub projection_freshness: u64,
    pub invariant_reports: u64,
    pub invariant_violations: u64,
    pub state_transitions: u64,
    pub artifact_refs: u64,
    pub manifests: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkspaceStateTransitionStatus {
    pub sequence_number: u64,
    pub from_state: String,
    pub to_state: String,
    pub trigger: String,
    pub message: String,
    pub transition_at: String,
    pub details: Value,
}

pub struct WorkspaceStore {
    path: PathBuf,
    conn: Connection,
}

pub struct WorkspaceView<'a> {
    store: &'a WorkspaceStore,
}

#[derive(Clone, Debug, Default)]
struct RunHealthCounts {
    resolved_run_configs: u64,
    container_contract_snapshots: u64,
    prompt_program_snapshots: u64,
    dataset_snapshots: u64,
    run_limits: u64,
    rendered_optimizer_states: u64,
    runtime_effects: u64,
    runtime_effect_admissions: u64,
    budget_reservations: u64,
    budget_commits: u64,
    budget_releases: u64,
    active_budget_reservations: u64,
    candidates: u64,
    parented_candidates: u64,
    train_frontier_candidates: u64,
    candidate_payloads: u64,
    candidate_deltas: u64,
    acceptance_decisions: u64,
    frontier_cells: u64,
    plan_links: u64,
    cache_profiles: u64,
    cache_accesses: u64,
    materializations: u64,
    evaluation_cache: u64,
    evaluation_cache_expected: u64,
    cache_profile_access_total: u64,
    event_stream_events: u64,
    usage_ledger: u64,
    usage_ledger_expected: u64,
    stopper_states: u64,
    checkpoints: u64,
    rollout_jobs: u64,
    sensor_frames: u64,
    rollouts: u64,
    rollout_events: u64,
    objective_sets: u64,
    objectives: u64,
    scores: u64,
    score_vector_sources: u64,
    score_vectors: u64,
    pareto_comparisons: u64,
    trace_annotations: u64,
    evidence_frames: u64,
    verifier_jobs: u64,
    subagent_invocations: u64,
    manifests: u64,
    state_transitions: u64,
    active_resource_leases: u64,
}

impl WorkspaceStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| OptimizerError::io(parent, source))?;
        }
        let conn = Connection::open(&path)?;
        let store = Self { path, conn };
        store.initialize()?;
        Ok(store)
    }

    pub fn open_existing(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Err(OptimizerError::Config(format!(
                "workspace database does not exist: {}",
                path.display()
            )));
        }
        let conn = Connection::open(&path)?;
        let store = Self { path, conn };
        store.initialize()?;
        Ok(store)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn view(&self) -> WorkspaceView<'_> {
        WorkspaceView { store: self }
    }

    pub fn status(&self) -> Result<WorkspaceStatus> {
        self.refresh_workspace_health()?;
        let schema_version = self
            .conn
            .query_row(
                "SELECT value FROM workspace_metadata WHERE key = 'schema_version'",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap_or_else(|_| "unknown".to_string());
        let mut stmt = self.conn.prepare(
            r#"
            SELECT run_id, state, cache_mode, cache_namespace, output_dir,
                   run_dir, manifest_path, best_candidate_id, cost_usd,
                   usage_json, started_at, completed_at, updated_at
            FROM optimization_runs
            ORDER BY started_at, run_id
            "#,
        )?;
        let mut rows = stmt.query([])?;
        let mut runs = Vec::new();
        while let Some(row) = rows.next()? {
            let run_id: String = row.get(0)?;
            let usage_json = row.get::<_, Option<String>>(9)?;
            runs.push(WorkspaceRunStatus {
                counts: self.entity_counts(&run_id)?,
                candidate_status_counts: self.group_counts(
                    "SELECT status, COUNT(*) FROM candidates WHERE run_id = ?1 GROUP BY status",
                    &run_id,
                )?,
                candidate_delta_operation_counts: self.group_counts(
                    "SELECT operation_kind, COUNT(*) FROM candidate_deltas WHERE run_id = ?1 GROUP BY operation_kind",
                    &run_id,
                )?,
                acceptance_decision_counts: self.group_counts(
                    "SELECT decision, COUNT(*) FROM acceptance_decisions WHERE run_id = ?1 GROUP BY decision",
                    &run_id,
                )?,
                plan_link_relation_counts: self.group_counts(
                    "SELECT relation, COUNT(*) FROM plan_links WHERE run_id = ?1 GROUP BY relation",
                    &run_id,
                )?,
                frontier_cell_status_counts: self.group_counts(
                    "SELECT status, COUNT(*) FROM frontier_cells WHERE run_id = ?1 GROUP BY status",
                    &run_id,
                )?,
                cache_access_action_counts: self.group_counts(
                    "SELECT action, COUNT(*) FROM cache_accesses WHERE run_id = ?1 GROUP BY action",
                    &run_id,
                )?,
                cache_access_boundary_counts: self.group_counts(
                    "SELECT boundary, COUNT(*) FROM cache_accesses WHERE run_id = ?1 GROUP BY boundary",
                    &run_id,
                )?,
                event_stream_event_type_counts: self.group_counts(
                    "SELECT event_type, COUNT(*) FROM event_stream_events WHERE run_id = ?1 GROUP BY event_type",
                    &run_id,
                )?,
                usage_ledger_boundary_counts: self.group_counts(
                    "SELECT boundary, COUNT(*) FROM usage_ledger WHERE run_id = ?1 GROUP BY boundary",
                    &run_id,
                )?,
                stopper_state_status_counts: self.group_counts(
                    "SELECT status, COUNT(*) FROM stopper_states WHERE run_id = ?1 GROUP BY status",
                    &run_id,
                )?,
                checkpoint_status_counts: self.group_counts(
                    "SELECT status, COUNT(*) FROM checkpoints WHERE run_id = ?1 GROUP BY status",
                    &run_id,
                )?,
                checkpoint_kind_counts: self.group_counts(
                    "SELECT checkpoint_kind, COUNT(*) FROM checkpoints WHERE run_id = ?1 GROUP BY checkpoint_kind",
                    &run_id,
                )?,
                optimizer_job_status_counts: self.group_counts(
                    "SELECT status, COUNT(*) FROM optimizer_jobs WHERE run_id = ?1 GROUP BY status",
                    &run_id,
                )?,
                optimizer_job_kind_counts: self.group_counts(
                    "SELECT kind, COUNT(*) FROM optimizer_jobs WHERE run_id = ?1 GROUP BY kind",
                    &run_id,
                )?,
                optimizer_job_lease_counts: self.group_counts(
                    r#"
                    SELECT
                        CASE
                            WHEN status IN ('completed', 'failed', 'cancelled', 'expired') THEN 'terminal'
                            WHEN lease_id IS NULL THEN 'unleased'
                            WHEN lease_expires_at IS NOT NULL AND lease_expires_at <= datetime('now') THEN 'lease_expired'
                            ELSE 'lease_active'
                        END AS lease_state,
                        COUNT(*)
                    FROM optimizer_jobs
                    WHERE run_id = ?1
                    GROUP BY lease_state
                    "#,
                    &run_id,
                )?,
                rollout_job_status_counts: self.group_counts(
                    "SELECT status, COUNT(*) FROM rollout_jobs WHERE run_id = ?1 GROUP BY status",
                    &run_id,
                )?,
                rollout_stage_counts: self.group_counts(
                    "SELECT evaluation_stage, COUNT(*) FROM rollout_jobs WHERE run_id = ?1 GROUP BY evaluation_stage",
                    &run_id,
                )?,
                rollout_status_counts: self.group_counts(
                    "SELECT status, COUNT(*) FROM rollouts WHERE run_id = ?1 GROUP BY status",
                    &run_id,
                )?,
                rollout_record_stage_counts: self.group_counts(
                    "SELECT evaluation_stage, COUNT(*) FROM rollouts WHERE run_id = ?1 GROUP BY evaluation_stage",
                    &run_id,
                )?,
                rollout_event_type_counts: self.group_counts(
                    "SELECT event_type, COUNT(*) FROM rollout_events WHERE run_id = ?1 GROUP BY event_type",
                    &run_id,
                )?,
                operation_status_counts: self.group_counts(
                    "SELECT status, COUNT(*) FROM operations WHERE run_id = ?1 GROUP BY status",
                    &run_id,
                )?,
                resource_lease_status_counts: self.group_counts(
                    "SELECT status, COUNT(*) FROM resource_leases WHERE run_id = ?1 GROUP BY status",
                    &run_id,
                )?,
                resource_lease_kind_counts: self.group_counts(
                    "SELECT resource_kind, COUNT(*) FROM resource_leases WHERE run_id = ?1 GROUP BY resource_kind",
                    &run_id,
                )?,
                score_objective_counts: self.group_counts(
                    "SELECT objective, COUNT(*) FROM scores WHERE run_id = ?1 GROUP BY objective",
                    &run_id,
                )?,
                score_stage_counts: self.group_counts(
                    "SELECT evaluation_stage, COUNT(*) FROM scores WHERE run_id = ?1 GROUP BY evaluation_stage",
                    &run_id,
                )?,
                trace_annotation_status_counts: self.group_counts(
                    "SELECT status, COUNT(*) FROM trace_annotations WHERE run_id = ?1 GROUP BY status",
                    &run_id,
                )?,
                evidence_frame_kind_counts: self.group_counts(
                    "SELECT kind, COUNT(*) FROM evidence_frames WHERE run_id = ?1 GROUP BY kind",
                    &run_id,
                )?,
                verifier_job_status_counts: self.group_counts(
                    "SELECT status, COUNT(*) FROM verifier_jobs WHERE run_id = ?1 GROUP BY status",
                    &run_id,
                )?,
                subagent_status_counts: self.group_counts(
                    "SELECT status, COUNT(*) FROM subagent_invocations WHERE run_id = ?1 GROUP BY status",
                    &run_id,
                )?,
                projection_freshness: self.projection_freshness_for_run(&run_id)?,
                invariant_report: self.invariant_report_for_run(&run_id)?,
                invariant_violation_severity_counts: self.group_counts(
                    "SELECT severity, COUNT(*) FROM invariant_violations WHERE run_id = ?1 GROUP BY severity",
                    &run_id,
                )?,
                latest_transition: self.latest_transition(&run_id)?,
                latest_rendered_state: self.latest_rendered_state(&run_id)?,
                rendered_state_phase_counts: self.group_counts(
                    "SELECT run_phase, COUNT(*) FROM rendered_optimizer_states WHERE run_id = ?1 GROUP BY run_phase",
                    &run_id,
                )?,
                budget_status: self.rendered_budget_status(&run_id)?,
                manifest_present: self.count_where("manifests", &run_id)? > 0,
                runtime_effect_status_counts: self.group_counts(
                    "SELECT status, COUNT(*) FROM runtime_effects WHERE run_id = ?1 GROUP BY status",
                    &run_id,
                )?,
                runtime_effect_lane_counts: self.group_counts(
                    "SELECT lane, COUNT(*) FROM runtime_effects WHERE run_id = ?1 GROUP BY lane",
                    &run_id,
                )?,
                runtime_effect_failure_class_counts: self.group_counts(
                    "SELECT failure_class, COUNT(*) FROM runtime_effects WHERE run_id = ?1 AND failure_class IS NOT NULL GROUP BY failure_class",
                    &run_id,
                )?,
                runtime_effect_admission_status_counts: self.group_counts(
                    "SELECT status, COUNT(*) FROM runtime_effect_admissions WHERE run_id = ?1 GROUP BY status",
                    &run_id,
                )?,
                budget_reservation_status_counts: self.group_counts(
                    "SELECT status, COUNT(*) FROM budget_reservations WHERE run_id = ?1 GROUP BY status",
                    &run_id,
                )?,
                budget_releases: self.count_where("budget_releases", &run_id)?,
                run_id,
                state: row.get(1)?,
                cache_mode: row.get(2)?,
                cache_namespace: row.get(3)?,
                output_dir: row.get(4)?,
                run_dir: row.get(5)?,
                manifest_path: row.get(6)?,
                best_candidate_id: row.get(7)?,
                cost_usd: row.get(8)?,
                usage: parse_json_or_null(usage_json.as_deref()),
                started_at: row.get(10)?,
                completed_at: row.get(11)?,
                updated_at: row.get(12)?,
            });
        }
        Ok(WorkspaceStatus {
            schema_version,
            path: self.path.display().to_string(),
            run_requests: self.run_requests()?,
            run_request_status_counts: self
                .global_group_counts("SELECT status, COUNT(*) FROM run_requests GROUP BY status")?,
            operation_status_counts: self
                .global_group_counts("SELECT status, COUNT(*) FROM operations GROUP BY status")?,
            resource_lease_status_counts: self.global_group_counts(
                "SELECT status, COUNT(*) FROM resource_leases GROUP BY status",
            )?,
            resource_lease_kind_counts: self.global_group_counts(
                "SELECT resource_kind, COUNT(*) FROM resource_leases GROUP BY resource_kind",
            )?,
            projection_status_counts: self.global_group_counts(
                "SELECT status, COUNT(*) FROM projection_freshness GROUP BY status",
            )?,
            invariant_status_counts: self.global_group_counts(
                "SELECT status, COUNT(*) FROM invariant_reports GROUP BY status",
            )?,
            invariant_violation_severity_counts: self.global_group_counts(
                "SELECT severity, COUNT(*) FROM invariant_violations GROUP BY severity",
            )?,
            workspace_projection_freshness: self.projection_freshness_for_run("__workspace__")?,
            workspace_invariant_report: self.invariant_report_for_run("__workspace__")?,
            runs,
        })
    }

    pub fn submit_run_request(
        &self,
        config_path: impl AsRef<Path>,
        priority: i64,
    ) -> Result<WorkspaceRunRequestStatus> {
        let config_path = config_path.as_ref();
        let config = SynthOptimizerConfig::from_toml_file(config_path)?;
        let paths = ArtifactPaths::new(&config.run.output_dir, &config.run.run_id);
        let cache_mode = CacheMode::from(config.cache.mode);
        let cache_namespace = config
            .cache
            .namespace
            .clone()
            .unwrap_or_else(|| format!("gepa:{}", config.run.run_id));
        let request_id = format!("runreq_{}", uuid::Uuid::new_v4().simple());
        let config_json = stable_json(&serde_json::to_value(&config)?);
        self.conn.execute(
            r#"
            INSERT INTO run_requests(
                request_id, run_id, status, config_path, config_json,
                container_url, cache_mode, cache_namespace, output_dir,
                run_dir, priority, submitted_at, updated_at
            ) VALUES (
                ?1, ?2, 'queued', ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10,
                datetime('now'), datetime('now')
            )
            "#,
            params![
                request_id,
                config.run.run_id,
                config_path.display().to_string(),
                config_json,
                config.container.url.unwrap_or_default(),
                cache_mode.as_str(),
                cache_namespace,
                config.run.output_dir.display().to_string(),
                paths.run_dir.display().to_string(),
                priority,
            ],
        )?;
        let request = self.run_request(&request_id)?;
        let mut metadata = Map::new();
        metadata.insert(
            "config_path".to_string(),
            Value::String(config_path.display().to_string()),
        );
        self.record_run_request_operation(
            "submit_run_request",
            &request,
            "completed",
            Some(format!(
                "submit:{}:{}",
                request.run_id,
                request.config_path.as_str()
            )),
            metadata,
        )?;
        Ok(request)
    }

    pub fn claim_next_run_request(
        &self,
        lease_id: &str,
        worker_id: Option<&str>,
        lease_seconds: u64,
    ) -> Result<Option<WorkspaceRunRequestStatus>> {
        self.recover_expired_run_requests()?;
        let request_id = self
            .conn
            .query_row(
                r#"
                SELECT queued.request_id
                FROM run_requests AS queued
                WHERE queued.status = 'queued'
                  AND NOT EXISTS (
                    SELECT 1
                    FROM run_requests AS active
                    WHERE active.request_id != queued.request_id
                      AND active.status IN ('leased', 'running')
                      AND (
                        active.lease_expires_at IS NULL
                        OR active.lease_expires_at > datetime('now')
                      )
                      AND (
                        (
                          active.container_url != ''
                          AND active.container_url = queued.container_url
                        )
                        OR (
                          active.cache_namespace != ''
                          AND active.cache_namespace = queued.cache_namespace
                        )
                      )
                  )
                ORDER BY queued.priority DESC, queued.submitted_at ASC, queued.request_id ASC
                LIMIT 1
                "#,
                [],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        let Some(request_id) = request_id else {
            return Ok(None);
        };
        let lease_modifier = format!("+{lease_seconds} seconds");
        let updated = self.conn.execute(
            r#"
            UPDATE run_requests
            SET status = 'leased',
                lease_id = ?1,
                worker_id = ?2,
                leased_at = datetime('now'),
                lease_expires_at = datetime('now', ?3),
                updated_at = datetime('now')
            WHERE request_id = ?4 AND status = 'queued'
            "#,
            params![lease_id, worker_id, lease_modifier, request_id],
        )?;
        if updated == 0 {
            return Ok(None);
        }
        let request = self.run_request(&request_id)?;
        let mut metadata = Map::new();
        if let Some(worker_id) = worker_id {
            metadata.insert(
                "worker_id".to_string(),
                Value::String(worker_id.to_string()),
            );
        }
        self.record_run_request_operation(
            "claim_run_request",
            &request,
            "completed",
            Some(lease_id.to_string()),
            metadata,
        )?;
        self.record_resource_leases_for_request(&request)?;
        Ok(Some(request))
    }

    pub fn heartbeat_run_request(
        &self,
        request_id: &str,
        lease_id: &str,
        lease_seconds: u64,
    ) -> Result<Option<WorkspaceRunRequestStatus>> {
        let lease_modifier = format!("+{lease_seconds} seconds");
        let updated = self.conn.execute(
            r#"
            UPDATE run_requests
            SET lease_expires_at = datetime('now', ?3),
                updated_at = datetime('now')
            WHERE request_id = ?1
              AND lease_id = ?2
              AND status IN ('leased', 'running')
              AND (
                lease_expires_at IS NULL
                OR lease_expires_at > datetime('now')
              )
            "#,
            params![request_id, lease_id, lease_modifier],
        )?;
        if updated == 0 {
            return Ok(None);
        }
        let request = self.run_request(request_id)?;
        self.record_run_request_operation(
            "heartbeat_run_request",
            &request,
            "completed",
            Some(lease_id.to_string()),
            Map::new(),
        )?;
        self.record_resource_leases_for_request(&request)?;
        Ok(Some(request))
    }

    pub fn recover_expired_run_requests(&self) -> Result<Vec<WorkspaceRunRequestStatus>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT request_id, status
            FROM run_requests
            WHERE status IN ('leased', 'running')
              AND lease_expires_at IS NOT NULL
              AND lease_expires_at <= datetime('now')
            ORDER BY lease_expires_at, request_id
            "#,
        )?;
        let mut rows = stmt.query([])?;
        let mut expired = Vec::new();
        while let Some(row) = rows.next()? {
            expired.push((row.get::<_, String>(0)?, row.get::<_, String>(1)?));
        }
        drop(rows);
        drop(stmt);

        let mut recovered = Vec::new();
        for (request_id, previous_status) in expired {
            let error = serde_json::json!({
                "reason_code": "lease_expired",
                "recovered_from_status": previous_status,
            });
            let updated = self.conn.execute(
                r#"
                UPDATE run_requests
                SET status = 'queued',
                    lease_id = NULL,
                    worker_id = NULL,
                    lease_expires_at = NULL,
                    started_at = CASE
                        WHEN status = 'running' THEN NULL
                        ELSE started_at
                    END,
                    error_json = ?2,
                    updated_at = datetime('now')
                WHERE request_id = ?1
                  AND status IN ('leased', 'running')
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at <= datetime('now')
                "#,
                params![request_id, stable_json(&error)],
            )?;
            if updated > 0 {
                let request = self.run_request(&request_id)?;
                self.update_resource_leases_for_request(&request_id, "expired")?;
                let mut metadata = Map::new();
                metadata.insert(
                    "recovered_from_status".to_string(),
                    Value::String(previous_status),
                );
                self.record_run_request_operation(
                    "recover_run_request",
                    &request,
                    "completed",
                    None,
                    metadata,
                )?;
                recovered.push(request);
            }
        }
        Ok(recovered)
    }

    pub fn run_request_status(&self, request_id: &str) -> Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT status FROM run_requests WHERE request_id = ?1",
                params![request_id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(OptimizerError::from)
    }

    pub fn run_request(&self, request_id: &str) -> Result<WorkspaceRunRequestStatus> {
        self.load_run_request(request_id)
    }

    pub fn run_request_config(&self, request_id: &str) -> Result<SynthOptimizerConfig> {
        let config_json: String = self.conn.query_row(
            "SELECT config_json FROM run_requests WHERE request_id = ?1",
            params![request_id],
            |row| row.get(0),
        )?;
        serde_json::from_str(&config_json).map_err(OptimizerError::from)
    }

    pub fn mark_run_request_started(&self, request_id: &str) -> Result<WorkspaceRunRequestStatus> {
        let request = self.update_run_request_status(
            request_id,
            "running",
            r#"
            UPDATE run_requests
            SET status = 'running',
                started_at = COALESCE(started_at, datetime('now')),
                updated_at = datetime('now')
            WHERE request_id = ?1
            "#,
            params![request_id],
        )?;
        self.record_run_request_operation(
            "start_run_request",
            &request,
            "completed",
            None,
            Map::new(),
        )?;
        Ok(request)
    }

    pub fn mark_run_request_started_for_lease(
        &self,
        request_id: &str,
        lease_id: &str,
        lease_seconds: u64,
    ) -> Result<Option<WorkspaceRunRequestStatus>> {
        let lease_modifier = format!("+{lease_seconds} seconds");
        let updated = self.conn.execute(
            r#"
            UPDATE run_requests
            SET status = 'running',
                started_at = COALESCE(started_at, datetime('now')),
                lease_expires_at = datetime('now', ?3),
                updated_at = datetime('now')
            WHERE request_id = ?1
              AND lease_id = ?2
              AND status = 'leased'
            "#,
            params![request_id, lease_id, lease_modifier],
        )?;
        if updated == 0 {
            return Ok(None);
        }
        let request = self.run_request(request_id)?;
        self.record_run_request_operation(
            "start_run_request",
            &request,
            "completed",
            Some(lease_id.to_string()),
            Map::new(),
        )?;
        self.record_resource_leases_for_request(&request)?;
        Ok(Some(request))
    }

    pub fn mark_run_request_completed(
        &self,
        request_id: &str,
    ) -> Result<WorkspaceRunRequestStatus> {
        let request = self.update_run_request_status(
            request_id,
            "completed",
            r#"
            UPDATE run_requests
            SET status = 'completed',
                finished_at = datetime('now'),
                lease_id = NULL,
                worker_id = NULL,
                lease_expires_at = NULL,
                error_json = NULL,
                updated_at = datetime('now')
            WHERE request_id = ?1
            "#,
            params![request_id],
        )?;
        self.update_resource_leases_for_request(request_id, "released")?;
        self.record_run_request_operation(
            "complete_run_request",
            &request,
            "completed",
            None,
            Map::new(),
        )?;
        Ok(request)
    }

    pub fn mark_run_request_completed_for_lease(
        &self,
        request_id: &str,
        lease_id: &str,
    ) -> Result<Option<WorkspaceRunRequestStatus>> {
        let updated = self.conn.execute(
            r#"
            UPDATE run_requests
            SET status = 'completed',
                finished_at = datetime('now'),
                lease_id = NULL,
                worker_id = NULL,
                lease_expires_at = NULL,
                error_json = NULL,
                updated_at = datetime('now')
            WHERE request_id = ?1
              AND lease_id = ?2
              AND status IN ('leased', 'running')
            "#,
            params![request_id, lease_id],
        )?;
        if updated == 0 {
            return Ok(None);
        }
        let request = self.run_request(request_id)?;
        self.update_resource_leases_for_request(request_id, "released")?;
        self.record_run_request_operation(
            "complete_run_request",
            &request,
            "completed",
            Some(lease_id.to_string()),
            Map::new(),
        )?;
        Ok(Some(request))
    }

    pub fn record_run_request_result(&self, request_id: &str, result: &Value) -> Result<()> {
        let best_candidate_id = result
            .get("best_candidate")
            .and_then(|candidate| candidate.get("candidate_id"))
            .and_then(Value::as_str);
        let usage = result.get("usage").unwrap_or(&Value::Null);
        self.conn.execute(
            r#"
            UPDATE run_requests
            SET run_workspace_db_path = ?2,
                result_manifest_path = ?3,
                best_candidate_id = ?4,
                cost_usd = ?5,
                usage_json = ?6,
                result_json = ?7,
                updated_at = datetime('now')
            WHERE request_id = ?1
            "#,
            params![
                request_id,
                result.get("workspace_db_path").and_then(Value::as_str),
                result.get("manifest_path").and_then(Value::as_str),
                best_candidate_id,
                result.get("cost_usd").and_then(Value::as_f64),
                stable_json(usage),
                stable_json(result),
            ],
        )?;
        Ok(())
    }

    pub fn record_run_request_result_for_lease(
        &self,
        request_id: &str,
        lease_id: &str,
        result: &Value,
    ) -> Result<bool> {
        let best_candidate_id = result
            .get("best_candidate")
            .and_then(|candidate| candidate.get("candidate_id"))
            .and_then(Value::as_str);
        let usage = result.get("usage").unwrap_or(&Value::Null);
        let updated = self.conn.execute(
            r#"
            UPDATE run_requests
            SET run_workspace_db_path = ?3,
                result_manifest_path = ?4,
                best_candidate_id = ?5,
                cost_usd = ?6,
                usage_json = ?7,
                result_json = ?8,
                updated_at = datetime('now')
            WHERE request_id = ?1
              AND lease_id = ?2
              AND status IN ('leased', 'running')
            "#,
            params![
                request_id,
                lease_id,
                result.get("workspace_db_path").and_then(Value::as_str),
                result.get("manifest_path").and_then(Value::as_str),
                best_candidate_id,
                result.get("cost_usd").and_then(Value::as_f64),
                stable_json(usage),
                stable_json(result),
            ],
        )?;
        Ok(updated > 0)
    }

    pub fn mark_run_request_failed(
        &self,
        request_id: &str,
        error: &Value,
    ) -> Result<WorkspaceRunRequestStatus> {
        let request = self.update_run_request_status(
            request_id,
            "failed",
            r#"
            UPDATE run_requests
            SET status = 'failed',
                finished_at = datetime('now'),
                lease_id = NULL,
                worker_id = NULL,
                lease_expires_at = NULL,
                error_json = ?2,
                updated_at = datetime('now')
            WHERE request_id = ?1
            "#,
            params![request_id, stable_json(error)],
        )?;
        self.update_resource_leases_for_request(request_id, "released")?;
        let mut metadata = Map::new();
        metadata.insert("error".to_string(), error.clone());
        self.record_run_request_operation(
            "fail_run_request",
            &request,
            "completed",
            None,
            metadata,
        )?;
        Ok(request)
    }

    pub fn mark_run_request_failed_for_lease(
        &self,
        request_id: &str,
        lease_id: &str,
        error: &Value,
    ) -> Result<Option<WorkspaceRunRequestStatus>> {
        let updated = self.conn.execute(
            r#"
            UPDATE run_requests
            SET status = 'failed',
                finished_at = datetime('now'),
                lease_id = NULL,
                worker_id = NULL,
                lease_expires_at = NULL,
                error_json = ?3,
                updated_at = datetime('now')
            WHERE request_id = ?1
              AND lease_id = ?2
              AND status IN ('leased', 'running')
            "#,
            params![request_id, lease_id, stable_json(error)],
        )?;
        if updated == 0 {
            return Ok(None);
        }
        let request = self.run_request(request_id)?;
        self.update_resource_leases_for_request(request_id, "released")?;
        let mut metadata = Map::new();
        metadata.insert("error".to_string(), error.clone());
        self.record_run_request_operation(
            "fail_run_request",
            &request,
            "completed",
            Some(lease_id.to_string()),
            metadata,
        )?;
        Ok(Some(request))
    }

    pub fn mark_run_request_cancelled(
        &self,
        request_id: &str,
        reason: &str,
    ) -> Result<WorkspaceRunRequestStatus> {
        let error = serde_json::json!({
            "reason": reason,
        });
        let request = self.update_run_request_status(
            request_id,
            "cancelled",
            r#"
            UPDATE run_requests
            SET status = 'cancelled',
                finished_at = datetime('now'),
                lease_id = NULL,
                worker_id = NULL,
                lease_expires_at = NULL,
                error_json = ?2,
                updated_at = datetime('now')
            WHERE request_id = ?1
            "#,
            params![request_id, stable_json(&error)],
        )?;
        self.update_resource_leases_for_request(request_id, "released")?;
        let mut metadata = Map::new();
        metadata.insert("reason".to_string(), Value::String(reason.to_string()));
        self.record_run_request_operation(
            "cancel_run_request",
            &request,
            "completed",
            None,
            metadata,
        )?;
        Ok(request)
    }

    pub fn mark_run_request_cancelled_for_lease(
        &self,
        request_id: &str,
        lease_id: &str,
        reason: &str,
    ) -> Result<Option<WorkspaceRunRequestStatus>> {
        let error = serde_json::json!({
            "reason": reason,
        });
        let updated = self.conn.execute(
            r#"
            UPDATE run_requests
            SET status = 'cancelled',
                finished_at = datetime('now'),
                lease_id = NULL,
                worker_id = NULL,
                lease_expires_at = NULL,
                error_json = ?3,
                updated_at = datetime('now')
            WHERE request_id = ?1
              AND lease_id = ?2
              AND status IN ('leased', 'running')
            "#,
            params![request_id, lease_id, stable_json(&error)],
        )?;
        if updated == 0 {
            return Ok(None);
        }
        let request = self.run_request(request_id)?;
        self.update_resource_leases_for_request(request_id, "released")?;
        let mut metadata = Map::new();
        metadata.insert("reason".to_string(), Value::String(reason.to_string()));
        self.record_run_request_operation(
            "cancel_run_request",
            &request,
            "completed",
            Some(lease_id.to_string()),
            metadata,
        )?;
        Ok(Some(request))
    }

    pub fn record_run_started(
        &self,
        paths: &ArtifactPaths,
        config: &SynthOptimizerConfig,
        cache_mode: CacheMode,
        cache_namespace: &str,
    ) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO optimization_runs(
                run_id, state, config_json, cache_mode, cache_namespace,
                output_dir, run_dir, manifest_path, started_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, datetime('now'), datetime('now'))
            ON CONFLICT(run_id) DO UPDATE SET
                state = optimization_runs.state,
                config_json = excluded.config_json,
                cache_mode = excluded.cache_mode,
                cache_namespace = excluded.cache_namespace,
                output_dir = excluded.output_dir,
                run_dir = excluded.run_dir,
                manifest_path = excluded.manifest_path,
                updated_at = datetime('now')
            "#,
            params![
                config.run.run_id,
                "created",
                stable_json(&serde_json::to_value(config)?),
                cache_mode.as_str(),
                cache_namespace,
                config.run.output_dir.display().to_string(),
                paths.run_dir.display().to_string(),
                paths.manifest_path.display().to_string(),
            ],
        )?;
        Ok(())
    }

    pub fn record_state_transition(
        &self,
        sequence_number: usize,
        transition: &OptimizerTransition,
    ) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO optimizer_state_history(
                run_id, sequence_number, from_state, to_state, trigger,
                message, transition_at, details_json
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            ON CONFLICT(run_id, sequence_number) DO UPDATE SET
                from_state = excluded.from_state,
                to_state = excluded.to_state,
                trigger = excluded.trigger,
                message = excluded.message,
                transition_at = excluded.transition_at,
                details_json = excluded.details_json
            "#,
            params![
                transition.run_id,
                sequence_number as i64,
                transition.from.as_str(),
                transition.to.as_str(),
                transition.trigger.as_str(),
                transition.message,
                transition.at,
                stable_json(&Value::Object(transition.details.clone())),
            ],
        )?;
        self.conn.execute(
            "UPDATE optimization_runs SET state = ?1, updated_at = datetime('now') WHERE run_id = ?2",
            params![transition.to.as_str(), transition.run_id],
        )?;
        let rendered = self.rendered_optimizer_state_for_transition(sequence_number, transition)?;
        self.record_rendered_optimizer_state(&rendered)?;
        Ok(())
    }

    pub fn record_rendered_optimizer_state(
        &self,
        record: &RenderedOptimizerStateRecord,
    ) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO rendered_optimizer_states(
                run_id, rendered_state_id, sequence_number, run_phase,
                generation_phase, candidate_phase, block_status, terminal_status,
                best_candidate_id, frontier_size, active_effect_count,
                active_job_count, queue_counts_json, budget_status_json,
                evidence_status_json, details_json, record_json, rendered_at,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, datetime('now'))
            ON CONFLICT(run_id, rendered_state_id) DO NOTHING
            "#,
            params![
                record.run_id,
                record.rendered_state_id,
                record.sequence_number as i64,
                record.run_phase,
                record.generation_phase.as_deref(),
                record.candidate_phase.as_deref(),
                record.block_status,
                record.terminal_status.as_deref(),
                record.best_candidate_id.as_deref(),
                record.frontier_size as i64,
                record.active_effect_count as i64,
                record.active_job_count as i64,
                stable_json(&record.queue_counts),
                stable_json(&record.budget_status),
                stable_json(&record.evidence_status),
                stable_json(&record.details),
                runtime_record_json(record),
                record.rendered_at,
            ],
        )?;
        Ok(())
    }

    pub fn record_resolved_run_config(&self, record: &ResolvedRunConfigRecord) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO resolved_run_configs(
                run_id, resolved_config_id, algorithm_id, config_hash,
                cache_mode, cache_namespace, output_dir, config_json,
                metadata_json, record_json, recorded_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, datetime('now'))
            ON CONFLICT(run_id, resolved_config_id) DO UPDATE SET
                algorithm_id = excluded.algorithm_id,
                config_hash = excluded.config_hash,
                cache_mode = excluded.cache_mode,
                cache_namespace = excluded.cache_namespace,
                output_dir = excluded.output_dir,
                config_json = excluded.config_json,
                metadata_json = excluded.metadata_json,
                record_json = excluded.record_json,
                recorded_at = excluded.recorded_at,
                updated_at = datetime('now')
            "#,
            params![
                record.run_id,
                record.resolved_config_id,
                record.algorithm_id,
                record.config_hash,
                record.cache_mode,
                record.cache_namespace,
                record.output_dir,
                stable_json(&record.config),
                stable_json(&Value::Object(record.metadata.clone())),
                runtime_record_json(record),
                record.recorded_at,
            ],
        )?;
        Ok(())
    }

    pub fn record_container_contract_snapshot(
        &self,
        record: &ContainerContractSnapshotRecord,
    ) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO container_contract_snapshots(
                run_id, contract_snapshot_id, container_url, contract_kind,
                contract_version, validation_status, capability_hash,
                metadata_response_json, health_response_json, metadata_json,
                record_json, recorded_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, datetime('now'))
            ON CONFLICT(run_id, contract_snapshot_id) DO UPDATE SET
                container_url = excluded.container_url,
                contract_kind = excluded.contract_kind,
                contract_version = excluded.contract_version,
                validation_status = excluded.validation_status,
                capability_hash = excluded.capability_hash,
                metadata_response_json = excluded.metadata_response_json,
                health_response_json = excluded.health_response_json,
                metadata_json = excluded.metadata_json,
                record_json = excluded.record_json,
                recorded_at = excluded.recorded_at,
                updated_at = datetime('now')
            "#,
            params![
                record.run_id,
                record.contract_snapshot_id,
                record.container_url,
                record.contract_kind,
                record.contract_version,
                record.validation_status,
                record.capability_hash,
                stable_json(&record.metadata_response),
                record.health_response.as_ref().map(stable_json),
                stable_json(&Value::Object(record.metadata.clone())),
                runtime_record_json(record),
                record.recorded_at,
            ],
        )?;
        Ok(())
    }

    pub fn record_prompt_program_snapshot(
        &self,
        record: &PromptProgramSnapshotRecord,
    ) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO prompt_program_snapshots(
                run_id, program_snapshot_id, program_id, program_hash,
                target_modules_json, mutable_field_ids_json, validation_status,
                program_json, metadata_json, record_json, recorded_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, datetime('now'))
            ON CONFLICT(run_id, program_snapshot_id) DO UPDATE SET
                program_id = excluded.program_id,
                program_hash = excluded.program_hash,
                target_modules_json = excluded.target_modules_json,
                mutable_field_ids_json = excluded.mutable_field_ids_json,
                validation_status = excluded.validation_status,
                program_json = excluded.program_json,
                metadata_json = excluded.metadata_json,
                record_json = excluded.record_json,
                recorded_at = excluded.recorded_at,
                updated_at = datetime('now')
            "#,
            params![
                record.run_id,
                record.program_snapshot_id,
                record.program_id,
                record.program_hash,
                stable_json(&serde_json::to_value(&record.target_modules)?),
                stable_json(&serde_json::to_value(&record.mutable_field_ids)?),
                record.validation_status,
                stable_json(&record.program),
                stable_json(&Value::Object(record.metadata.clone())),
                runtime_record_json(record),
                record.recorded_at,
            ],
        )?;
        Ok(())
    }

    pub fn record_dataset_snapshot(&self, record: &DatasetSnapshotRecord) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO dataset_snapshots(
                run_id, dataset_snapshot_id, dataset_id, split, row_count,
                seed_count, seeds_json, filters_json, rows_hash, rows_json,
                dataset_metadata_json, rows_metadata_json, metadata_json,
                record_json, recorded_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, datetime('now'))
            ON CONFLICT(run_id, dataset_snapshot_id) DO UPDATE SET
                dataset_id = excluded.dataset_id,
                split = excluded.split,
                row_count = excluded.row_count,
                seed_count = excluded.seed_count,
                seeds_json = excluded.seeds_json,
                filters_json = excluded.filters_json,
                rows_hash = excluded.rows_hash,
                rows_json = excluded.rows_json,
                dataset_metadata_json = excluded.dataset_metadata_json,
                rows_metadata_json = excluded.rows_metadata_json,
                metadata_json = excluded.metadata_json,
                record_json = excluded.record_json,
                recorded_at = excluded.recorded_at,
                updated_at = datetime('now')
            "#,
            params![
                record.run_id,
                record.dataset_snapshot_id,
                record.dataset_id,
                record.split,
                record.row_count as i64,
                record.seed_count as i64,
                stable_json(&serde_json::to_value(&record.seeds)?),
                stable_json(&record.filters),
                record.rows_hash,
                stable_json(&record.rows),
                stable_json(&record.dataset_metadata),
                stable_json(&record.rows_metadata),
                stable_json(&Value::Object(record.metadata.clone())),
                runtime_record_json(record),
                record.recorded_at,
            ],
        )?;
        Ok(())
    }

    pub fn record_run_limits(&self, record: &RunLimitsRecord) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO run_limits(
                run_id, run_limits_id, max_total_rollouts, max_cost_usd,
                max_time_seconds, max_prompt_tokens, max_completion_tokens,
                max_total_tokens, hard_limit, stop_policy, metadata_json,
                record_json, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, datetime('now'))
            ON CONFLICT(run_id, run_limits_id) DO UPDATE SET
                max_total_rollouts = excluded.max_total_rollouts,
                max_cost_usd = excluded.max_cost_usd,
                max_time_seconds = excluded.max_time_seconds,
                max_prompt_tokens = excluded.max_prompt_tokens,
                max_completion_tokens = excluded.max_completion_tokens,
                max_total_tokens = excluded.max_total_tokens,
                hard_limit = excluded.hard_limit,
                stop_policy = excluded.stop_policy,
                metadata_json = excluded.metadata_json,
                record_json = excluded.record_json,
                updated_at = datetime('now')
            "#,
            params![
                record.run_id,
                record.run_limits_id,
                record.max_total_rollouts.map(|value| value as i64),
                record.max_cost_usd,
                record.max_time_seconds.map(|value| value as i64),
                record.max_prompt_tokens.map(|value| value as i64),
                record.max_completion_tokens.map(|value| value as i64),
                record.max_total_tokens.map(|value| value as i64),
                if record.hard_limit { 1 } else { 0 },
                record.stop_policy,
                stable_json(&Value::Object(record.metadata.clone())),
                runtime_record_json(record),
            ],
        )?;
        Ok(())
    }

    pub fn record_runtime_effect(&self, record: &RuntimeEffectRecord) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO runtime_effects(
                run_id, runtime_effect_id, effect_kind, lane, status,
                subject_type, subject_id, idempotency_key, cache_key, job_id,
                budget_reservation_id, attempt, failure_class, payload_json,
                metadata_json, record_json, planned_at, updated_at, terminal_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, datetime('now'), ?18)
            ON CONFLICT(run_id, runtime_effect_id) DO UPDATE SET
                effect_kind = excluded.effect_kind,
                lane = excluded.lane,
                status = excluded.status,
                subject_type = excluded.subject_type,
                subject_id = excluded.subject_id,
                idempotency_key = excluded.idempotency_key,
                cache_key = excluded.cache_key,
                job_id = excluded.job_id,
                budget_reservation_id = excluded.budget_reservation_id,
                attempt = excluded.attempt,
                failure_class = excluded.failure_class,
                payload_json = excluded.payload_json,
                metadata_json = excluded.metadata_json,
                record_json = excluded.record_json,
                planned_at = COALESCE(runtime_effects.planned_at, excluded.planned_at),
                updated_at = datetime('now'),
                terminal_at = COALESCE(excluded.terminal_at, runtime_effects.terminal_at)
            "#,
            params![
                record.run_id,
                record.runtime_effect_id,
                record.effect_kind,
                record.lane,
                record.status,
                record.subject_type,
                record.subject_id,
                record.idempotency_key,
                record.cache_key,
                record.job_id,
                record.budget_reservation_id,
                record.attempt as i64,
                record.failure_class,
                stable_json(&record.payload),
                stable_json(&Value::Object(record.metadata.clone())),
                runtime_record_json(record),
                record.planned_at,
                record.terminal_at,
            ],
        )?;
        Ok(())
    }

    pub fn record_runtime_effect_admission(
        &self,
        record: &RuntimeEffectAdmissionRecord,
    ) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO runtime_effect_admissions(
                run_id, admission_id, runtime_effect_id, effect_kind, lane,
                subject_type, subject_id, idempotency_key, status,
                rejection_reason, max_cost_usd, max_prompt_tokens,
                max_completion_tokens, max_total_tokens, max_rollouts,
                max_wall_seconds, ledger_json, metadata_json, record_json,
                checked_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20)
            ON CONFLICT(run_id, admission_id) DO UPDATE SET
                runtime_effect_id = excluded.runtime_effect_id,
                effect_kind = excluded.effect_kind,
                lane = excluded.lane,
                subject_type = excluded.subject_type,
                subject_id = excluded.subject_id,
                idempotency_key = excluded.idempotency_key,
                status = excluded.status,
                rejection_reason = excluded.rejection_reason,
                max_cost_usd = excluded.max_cost_usd,
                max_prompt_tokens = excluded.max_prompt_tokens,
                max_completion_tokens = excluded.max_completion_tokens,
                max_total_tokens = excluded.max_total_tokens,
                max_rollouts = excluded.max_rollouts,
                max_wall_seconds = excluded.max_wall_seconds,
                ledger_json = excluded.ledger_json,
                metadata_json = excluded.metadata_json,
                record_json = excluded.record_json,
                checked_at = excluded.checked_at
            "#,
            params![
                record.run_id,
                record.admission_id,
                record.runtime_effect_id,
                record.effect_kind,
                record.lane,
                record.subject_type,
                record.subject_id,
                record.idempotency_key,
                record.status,
                record.rejection_reason,
                record.max_cost_usd,
                record.max_prompt_tokens.map(|value| value as i64),
                record.max_completion_tokens.map(|value| value as i64),
                record.max_total_tokens.map(|value| value as i64),
                record.max_rollouts.map(|value| value as i64),
                record.max_wall_seconds.map(|value| value as i64),
                stable_json(&serde_json::to_value(&record.ledger)?),
                stable_json(&Value::Object(record.metadata.clone())),
                runtime_record_json(record),
                record.checked_at,
            ],
        )?;
        Ok(())
    }

    pub fn record_optimizer_job(&self, job: &OptimizerJob) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO optimizer_jobs(
                run_id, job_id, kind, status, candidate_id, sensor_frame_id,
                attempt, lease_id, worker_id, leased_at, lease_expires_at,
                heartbeat_at, next_retry_at, retry_policy_json, failure_json,
                payload_json, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, NULL, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, datetime('now'))
            ON CONFLICT(run_id, job_id) DO UPDATE SET
                kind = excluded.kind,
                status = excluded.status,
                candidate_id = excluded.candidate_id,
                attempt = excluded.attempt,
                lease_id = excluded.lease_id,
                worker_id = excluded.worker_id,
                leased_at = excluded.leased_at,
                lease_expires_at = excluded.lease_expires_at,
                heartbeat_at = excluded.heartbeat_at,
                next_retry_at = excluded.next_retry_at,
                retry_policy_json = excluded.retry_policy_json,
                failure_json = excluded.failure_json,
                payload_json = excluded.payload_json,
                updated_at = datetime('now')
            "#,
            params![
                job.run_id,
                job.job_id,
                job.kind.as_str(),
                job.status.as_str(),
                job.candidate_id,
                job.attempt as i64,
                job.lease_id,
                job.worker_id,
                job.leased_at,
                job.lease_expires_at,
                job.heartbeat_at,
                job.next_retry_at,
                stable_json(&serde_json::to_value(&job.retry_policy)?),
                job.failure
                    .as_ref()
                    .map(serde_json::to_value)
                    .transpose()?
                    .as_ref()
                    .map(stable_json),
                stable_json(&Value::Object(job.payload.clone())),
            ],
        )?;
        Ok(())
    }

    pub fn claim_next_optimizer_job(
        &self,
        run_id: &str,
        lease_id: &str,
        worker_id: Option<&str>,
        lease_seconds: u64,
    ) -> Result<Option<OptimizerJob>> {
        self.recover_expired_optimizer_jobs(run_id)?;
        let job_id = self
            .conn
            .query_row(
                r#"
                SELECT job_id
                FROM optimizer_jobs
                WHERE run_id = ?1
                  AND (
                    status = 'pending'
                    OR (
                        status = 'retry_scheduled'
                        AND (next_retry_at IS NULL OR next_retry_at <= datetime('now'))
                    )
                  )
                ORDER BY updated_at ASC, job_id ASC
                LIMIT 1
                "#,
                params![run_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?;
        let Some(job_id) = job_id else {
            return Ok(None);
        };
        let lease_modifier = format!("+{lease_seconds} seconds");
        let updated = self.conn.execute(
            r#"
            UPDATE optimizer_jobs
            SET status = 'leased',
                lease_id = ?3,
                worker_id = ?4,
                leased_at = datetime('now'),
                lease_expires_at = datetime('now', ?5),
                heartbeat_at = datetime('now'),
                next_retry_at = NULL,
                attempt = attempt + 1,
                updated_at = datetime('now')
            WHERE run_id = ?1
              AND job_id = ?2
              AND (
                status = 'pending'
                OR (
                    status = 'retry_scheduled'
                    AND (next_retry_at IS NULL OR next_retry_at <= datetime('now'))
                )
              )
            "#,
            params![run_id, job_id, lease_id, worker_id, lease_modifier],
        )?;
        if updated == 0 {
            return Ok(None);
        }
        self.optimizer_job(run_id, &job_id).map(Some)
    }

    pub fn claim_optimizer_job(
        &self,
        run_id: &str,
        job_id: &str,
        lease_id: &str,
        worker_id: Option<&str>,
        lease_seconds: u64,
    ) -> Result<Option<OptimizerJob>> {
        self.recover_expired_optimizer_jobs(run_id)?;
        let lease_modifier = format!("+{lease_seconds} seconds");
        let updated = self.conn.execute(
            r#"
            UPDATE optimizer_jobs
            SET status = 'leased',
                lease_id = ?3,
                worker_id = ?4,
                leased_at = datetime('now'),
                lease_expires_at = datetime('now', ?5),
                heartbeat_at = datetime('now'),
                next_retry_at = NULL,
                attempt = attempt + 1,
                updated_at = datetime('now')
            WHERE run_id = ?1
              AND job_id = ?2
              AND (
                status = 'pending'
                OR (
                    status = 'retry_scheduled'
                    AND (next_retry_at IS NULL OR next_retry_at <= datetime('now'))
                )
              )
            "#,
            params![run_id, job_id, lease_id, worker_id, lease_modifier],
        )?;
        if updated == 0 {
            return Ok(None);
        }
        self.optimizer_job(run_id, job_id).map(Some)
    }

    pub fn mark_optimizer_job_running(
        &self,
        run_id: &str,
        job_id: &str,
        lease_id: &str,
        lease_seconds: u64,
    ) -> Result<Option<OptimizerJob>> {
        let lease_modifier = format!("+{lease_seconds} seconds");
        let updated = self.conn.execute(
            r#"
            UPDATE optimizer_jobs
            SET status = 'running',
                heartbeat_at = datetime('now'),
                lease_expires_at = datetime('now', ?4),
                updated_at = datetime('now')
            WHERE run_id = ?1
              AND job_id = ?2
              AND lease_id = ?3
              AND status IN ('leased', 'running', 'annotating', 'verifying')
            "#,
            params![run_id, job_id, lease_id, lease_modifier],
        )?;
        if updated == 0 {
            return Ok(None);
        }
        self.optimizer_job(run_id, job_id).map(Some)
    }

    pub fn optimizer_job_claimable(&self, run_id: &str, job_id: &str) -> Result<bool> {
        let count = self.conn.query_row(
            r#"
            SELECT COUNT(*)
            FROM optimizer_jobs
            WHERE run_id = ?1
              AND job_id = ?2
              AND (
                status = 'pending'
                OR (
                    status = 'retry_scheduled'
                    AND (next_retry_at IS NULL OR next_retry_at <= datetime('now'))
                )
              )
            "#,
            params![run_id, job_id],
            |row| row.get::<_, i64>(0),
        )?;
        Ok(count > 0)
    }

    pub fn schedule_optimizer_job_retry(
        &self,
        run_id: &str,
        job_id: &str,
        lease_id: &str,
        backoff_seconds: u64,
        failure: &crate::failures::FailurePayload,
    ) -> Result<Option<OptimizerJob>> {
        let retry_modifier = format!("+{backoff_seconds} seconds");
        let failure_json = stable_json(&serde_json::to_value(failure)?);
        let updated = self.conn.execute(
            r#"
            UPDATE optimizer_jobs
            SET status = 'retry_scheduled',
                lease_id = NULL,
                worker_id = NULL,
                leased_at = NULL,
                lease_expires_at = NULL,
                heartbeat_at = NULL,
                next_retry_at = datetime('now', ?4),
                failure_json = ?5,
                updated_at = datetime('now')
            WHERE run_id = ?1
              AND job_id = ?2
              AND lease_id = ?3
              AND status IN ('leased', 'running')
            "#,
            params![run_id, job_id, lease_id, retry_modifier, failure_json],
        )?;
        if updated == 0 {
            return Ok(None);
        }
        self.optimizer_job(run_id, job_id).map(Some)
    }

    pub fn heartbeat_optimizer_job(
        &self,
        run_id: &str,
        job_id: &str,
        lease_id: &str,
        lease_seconds: u64,
    ) -> Result<Option<OptimizerJob>> {
        let lease_modifier = format!("+{lease_seconds} seconds");
        let updated = self.conn.execute(
            r#"
            UPDATE optimizer_jobs
            SET heartbeat_at = datetime('now'),
                lease_expires_at = datetime('now', ?4),
                updated_at = datetime('now')
            WHERE run_id = ?1
              AND job_id = ?2
              AND lease_id = ?3
              AND status IN ('leased', 'running', 'annotating', 'verifying')
            "#,
            params![run_id, job_id, lease_id, lease_modifier],
        )?;
        if updated == 0 {
            return Ok(None);
        }
        self.optimizer_job(run_id, job_id).map(Some)
    }

    pub fn recover_expired_optimizer_jobs(&self, run_id: &str) -> Result<Vec<OptimizerJob>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT job_id, attempt, retry_policy_json
            FROM optimizer_jobs
            WHERE run_id = ?1
              AND status IN ('leased', 'running', 'annotating', 'verifying')
              AND lease_expires_at IS NOT NULL
              AND lease_expires_at <= datetime('now')
            ORDER BY lease_expires_at, job_id
            "#,
        )?;
        let mut rows = stmt.query(params![run_id])?;
        let mut expired = Vec::new();
        while let Some(row) = rows.next()? {
            let retry_policy_json: String = row.get(2)?;
            let retry_policy: RetryPolicy = serde_json::from_str(&retry_policy_json)?;
            expired.push((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?.max(0) as u32,
                retry_policy,
            ));
        }
        drop(rows);
        drop(stmt);

        let mut recovered = Vec::new();
        for (job_id, attempt, retry_policy) in expired {
            let retryable = attempt < retry_policy.max_attempts;
            let status = if retryable {
                "retry_scheduled"
            } else {
                "expired"
            };
            let retry_modifier = format!("+{} seconds", retry_policy.backoff_seconds);
            let updated = self.conn.execute(
                r#"
                UPDATE optimizer_jobs
                SET status = ?3,
                    lease_id = NULL,
                    worker_id = NULL,
                    lease_expires_at = NULL,
                    heartbeat_at = NULL,
                    next_retry_at = CASE
                        WHEN ?4 THEN datetime('now', ?5)
                        ELSE NULL
                    END,
                    updated_at = datetime('now')
                WHERE run_id = ?1
                  AND job_id = ?2
                  AND status IN ('leased', 'running', 'annotating', 'verifying')
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at <= datetime('now')
                "#,
                params![run_id, job_id, status, retryable, retry_modifier],
            )?;
            if updated > 0 {
                recovered.push(self.optimizer_job(run_id, &job_id)?);
            }
        }
        Ok(recovered)
    }

    pub fn optimizer_job(&self, run_id: &str, job_id: &str) -> Result<OptimizerJob> {
        self.maybe_optimizer_job(run_id, job_id)?.ok_or_else(|| {
            OptimizerError::Config(format!(
                "optimizer job does not exist run_id={run_id} job_id={job_id}"
            ))
        })
    }

    pub fn maybe_optimizer_job(&self, run_id: &str, job_id: &str) -> Result<Option<OptimizerJob>> {
        self.conn
            .query_row(
                r#"
                SELECT job_id, run_id, kind, status, candidate_id, attempt,
                       lease_id, worker_id, leased_at, lease_expires_at,
                       heartbeat_at, next_retry_at, retry_policy_json,
                       failure_json, payload_json
                FROM optimizer_jobs
                WHERE run_id = ?1 AND job_id = ?2
                "#,
                params![run_id, job_id],
                optimizer_job_from_row,
            )
            .optional()
            .map_err(OptimizerError::from)
    }

    pub fn runtime_effect(
        &self,
        run_id: &str,
        runtime_effect_id: &str,
    ) -> Result<RuntimeEffectRecord> {
        self.conn
            .query_row(
                r#"
                SELECT record_json
                FROM runtime_effects
                WHERE run_id = ?1 AND runtime_effect_id = ?2
                "#,
                params![run_id, runtime_effect_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .map(|raw| serde_json::from_str(&raw).map_err(OptimizerError::from))
            .transpose()?
            .ok_or_else(|| {
                OptimizerError::Invariant(format!(
                    "runtime effect does not exist run_id={run_id} runtime_effect_id={runtime_effect_id}"
                ))
            })
    }

    pub fn budget_reservation(
        &self,
        run_id: &str,
        budget_reservation_id: &str,
    ) -> Result<BudgetReservationRecord> {
        self.conn
            .query_row(
                r#"
                SELECT record_json
                FROM budget_reservations
                WHERE run_id = ?1 AND budget_reservation_id = ?2
                "#,
                params![run_id, budget_reservation_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .map(|raw| serde_json::from_str(&raw).map_err(OptimizerError::from))
            .transpose()?
            .ok_or_else(|| {
                OptimizerError::Invariant(format!(
                    "budget reservation does not exist run_id={run_id} budget_reservation_id={budget_reservation_id}"
                ))
            })
    }

    pub fn record_budget_reservation(&self, record: &BudgetReservationRecord) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO budget_reservations(
                run_id, budget_reservation_id, runtime_effect_id, status,
                max_cost_usd, max_prompt_tokens, max_completion_tokens,
                max_total_tokens, max_rollouts, max_wall_seconds,
                metadata_json, record_json, reserved_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, datetime('now'))
            ON CONFLICT(run_id, budget_reservation_id) DO UPDATE SET
                runtime_effect_id = excluded.runtime_effect_id,
                status = excluded.status,
                max_cost_usd = excluded.max_cost_usd,
                max_prompt_tokens = excluded.max_prompt_tokens,
                max_completion_tokens = excluded.max_completion_tokens,
                max_total_tokens = excluded.max_total_tokens,
                max_rollouts = excluded.max_rollouts,
                max_wall_seconds = excluded.max_wall_seconds,
                metadata_json = excluded.metadata_json,
                record_json = excluded.record_json,
                reserved_at = COALESCE(budget_reservations.reserved_at, excluded.reserved_at),
                updated_at = datetime('now')
            "#,
            params![
                record.run_id,
                record.budget_reservation_id,
                record.runtime_effect_id,
                record.status,
                record.max_cost_usd,
                record.max_prompt_tokens.map(|value| value as i64),
                record.max_completion_tokens.map(|value| value as i64),
                record.max_total_tokens.map(|value| value as i64),
                record.max_rollouts.map(|value| value as i64),
                record.max_wall_seconds.map(|value| value as i64),
                stable_json(&Value::Object(record.metadata.clone())),
                runtime_record_json(record),
                record.reserved_at,
            ],
        )?;
        Ok(())
    }

    pub fn record_budget_commit(&self, record: &BudgetCommitRecord) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO budget_commits(
                run_id, budget_commit_id, runtime_effect_id,
                budget_reservation_id, cost_usd, prompt_tokens,
                completion_tokens, total_tokens, rollout_count, wall_seconds,
                metadata_json, record_json, committed_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
            ON CONFLICT(run_id, budget_commit_id) DO UPDATE SET
                runtime_effect_id = excluded.runtime_effect_id,
                budget_reservation_id = excluded.budget_reservation_id,
                cost_usd = excluded.cost_usd,
                prompt_tokens = excluded.prompt_tokens,
                completion_tokens = excluded.completion_tokens,
                total_tokens = excluded.total_tokens,
                rollout_count = excluded.rollout_count,
                wall_seconds = excluded.wall_seconds,
                metadata_json = excluded.metadata_json,
                record_json = excluded.record_json,
                committed_at = excluded.committed_at
            "#,
            params![
                record.run_id,
                record.budget_commit_id,
                record.runtime_effect_id,
                record.budget_reservation_id,
                record.cost_usd,
                record.prompt_tokens as i64,
                record.completion_tokens as i64,
                record.total_tokens as i64,
                record.rollout_count as i64,
                record.wall_seconds as i64,
                stable_json(&Value::Object(record.metadata.clone())),
                runtime_record_json(record),
                record.committed_at,
            ],
        )?;
        Ok(())
    }

    pub fn record_budget_release(&self, record: &BudgetReleaseRecord) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO budget_releases(
                run_id, budget_release_id, runtime_effect_id,
                budget_reservation_id, release_reason, released_cost_usd,
                released_prompt_tokens, released_completion_tokens,
                released_total_tokens, released_rollouts, released_wall_seconds,
                metadata_json, record_json, released_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
            ON CONFLICT(run_id, budget_release_id) DO UPDATE SET
                runtime_effect_id = excluded.runtime_effect_id,
                budget_reservation_id = excluded.budget_reservation_id,
                release_reason = excluded.release_reason,
                released_cost_usd = excluded.released_cost_usd,
                released_prompt_tokens = excluded.released_prompt_tokens,
                released_completion_tokens = excluded.released_completion_tokens,
                released_total_tokens = excluded.released_total_tokens,
                released_rollouts = excluded.released_rollouts,
                released_wall_seconds = excluded.released_wall_seconds,
                metadata_json = excluded.metadata_json,
                record_json = excluded.record_json,
                released_at = excluded.released_at
            "#,
            params![
                record.run_id,
                record.budget_release_id,
                record.runtime_effect_id,
                record.budget_reservation_id,
                record.release_reason,
                record.released_cost_usd,
                record.released_prompt_tokens as i64,
                record.released_completion_tokens as i64,
                record.released_total_tokens as i64,
                record.released_rollouts as i64,
                record.released_wall_seconds as i64,
                stable_json(&Value::Object(record.metadata.clone())),
                runtime_record_json(record),
                record.released_at,
            ],
        )?;
        Ok(())
    }

    pub fn latest_run_limits(&self, run_id: &str) -> Result<Option<RunLimitsRecord>> {
        self.conn
            .query_row(
                r#"
                SELECT record_json
                FROM run_limits
                WHERE run_id = ?1
                ORDER BY updated_at DESC, run_limits_id DESC
                LIMIT 1
                "#,
                params![run_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .map(|raw| serde_json::from_str(&raw).map_err(OptimizerError::from))
            .transpose()
    }

    pub fn required_run_limits(&self, run_id: &str) -> Result<RunLimitsRecord> {
        self.latest_run_limits(run_id)?.ok_or_else(|| {
            OptimizerError::Invariant(format!(
                "run_id={run_id} has no run_limits record before runtime effect admission"
            ))
        })
    }

    pub fn budget_ledger_snapshot(&self, run_id: &str) -> Result<BudgetLedgerSnapshot> {
        let limits = self.required_run_limits(run_id)?;
        let spent = self.conn.query_row(
            r#"
            SELECT COALESCE(SUM(cost_usd), 0.0),
                   COALESCE(SUM(prompt_tokens), 0),
                   COALESCE(SUM(completion_tokens), 0),
                   COALESCE(SUM(total_tokens), 0),
                   COALESCE(SUM(rollout_count), 0),
                   COALESCE(SUM(wall_seconds), 0)
            FROM budget_commits
            WHERE run_id = ?1
            "#,
            params![run_id],
            |row| {
                Ok(BudgetLedgerTotals {
                    cost_usd: row.get::<_, f64>(0)?,
                    prompt_tokens: nonnegative_u64(row.get::<_, i64>(1)?),
                    completion_tokens: nonnegative_u64(row.get::<_, i64>(2)?),
                    total_tokens: nonnegative_u64(row.get::<_, i64>(3)?),
                    rollouts: nonnegative_u64(row.get::<_, i64>(4)?),
                    wall_seconds: nonnegative_u64(row.get::<_, i64>(5)?),
                })
            },
        )?;
        let reserved = self.conn.query_row(
            r#"
            SELECT COALESCE(SUM(max_cost_usd), 0.0),
                   COALESCE(SUM(max_prompt_tokens), 0),
                   COALESCE(SUM(max_completion_tokens), 0),
                   COALESCE(SUM(max_total_tokens), 0),
                   COALESCE(SUM(max_rollouts), 0),
                   COALESCE(SUM(max_wall_seconds), 0)
            FROM budget_reservations
            WHERE run_id = ?1 AND status IN ('reserved', 'planned', 'active', 'leased', 'running')
            "#,
            params![run_id],
            |row| {
                Ok(BudgetLedgerTotals {
                    cost_usd: row.get::<_, f64>(0)?,
                    prompt_tokens: nonnegative_u64(row.get::<_, i64>(1)?),
                    completion_tokens: nonnegative_u64(row.get::<_, i64>(2)?),
                    total_tokens: nonnegative_u64(row.get::<_, i64>(3)?),
                    rollouts: nonnegative_u64(row.get::<_, i64>(4)?),
                    wall_seconds: nonnegative_u64(row.get::<_, i64>(5)?),
                })
            },
        )?;
        Ok(BudgetLedgerSnapshot::from_totals(
            run_id, &limits, spent, reserved,
        ))
    }

    pub fn persist_state_history(&self, transitions: &[OptimizerTransition]) -> Result<()> {
        for (index, transition) in transitions.iter().enumerate() {
            self.record_state_transition(index + 1, transition)?;
        }
        Ok(())
    }

    pub fn persist_candidate_registry(&mut self, run_id: &str, candidates: &[Value]) -> Result<()> {
        let tx = self.conn.transaction()?;
        for candidate in candidates {
            let candidate_id = required_string(candidate, "candidate_id")?;
            let parent_id = optional_string(candidate, "parent_id");
            let source = optional_string(candidate, "source").unwrap_or_default();
            let status = optional_string(candidate, "status").unwrap_or_default();
            tx.execute(
                r#"
                INSERT INTO candidates(
                    run_id, candidate_id, parent_id, source, status,
                    payload_json, lever_bundle_json, minibatch_reward,
                    train_reward, heldout_reward, record_json, updated_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, datetime('now'))
                ON CONFLICT(run_id, candidate_id) DO UPDATE SET
                    parent_id = excluded.parent_id,
                    source = excluded.source,
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    lever_bundle_json = excluded.lever_bundle_json,
                    minibatch_reward = excluded.minibatch_reward,
                    train_reward = excluded.train_reward,
                    heldout_reward = excluded.heldout_reward,
                    record_json = excluded.record_json,
                    updated_at = datetime('now')
                "#,
                params![
                    run_id,
                    candidate_id,
                    parent_id,
                    source,
                    status,
                    stable_json(candidate.get("payload").unwrap_or(&Value::Null)),
                    stable_json(candidate.get("lever_bundle").unwrap_or(&Value::Null)),
                    candidate.get("minibatch_reward").and_then(Value::as_f64),
                    candidate.get("train_reward").and_then(Value::as_f64),
                    candidate.get("heldout_reward").and_then(Value::as_f64),
                    stable_json(candidate),
                ],
            )?;
            let payload_value = candidate.get("payload").unwrap_or(&Value::Null);
            let lever_bundle_value = candidate.get("lever_bundle").unwrap_or(&Value::Null);
            let payload_record = CandidatePayloadRecord::from_input(CandidatePayloadInput {
                candidate_id: &candidate_id,
                parent_id: parent_id.clone(),
                source: &source,
                status: &status,
                payload: payload_value,
                lever_bundle: lever_bundle_value,
            });
            upsert_candidate_payload_tx(&tx, run_id, &payload_record)?;
            upsert_plan_link_tx(
                &tx,
                run_id,
                &PlanLinkRecord::from_input(PlanLinkInput {
                    source_type: "candidate",
                    source_id: &candidate_id,
                    target_type: "candidate_payload",
                    target_id: &payload_record.candidate_payload_id,
                    relation: "candidate_payload",
                    status: "active",
                    confidence: 1.0,
                    metadata: Map::new(),
                }),
            )?;
            let parent_snapshot = if let Some(parent_id) = parent_id.as_deref() {
                let snapshot = candidate_snapshot_tx(&tx, run_id, parent_id)?.unwrap_or_default();
                let delta = CandidateDeltaRecord::from_input(CandidateDeltaInput {
                    candidate_id: &candidate_id,
                    parent_candidate_id: parent_id,
                    source: &source,
                    status: &status,
                    parent_payload: &snapshot.payload,
                    parent_lever_bundle: &snapshot.lever_bundle,
                    child_payload: payload_value,
                    child_lever_bundle: lever_bundle_value,
                });
                upsert_candidate_delta_tx(&tx, run_id, &delta)?;
                upsert_plan_link_tx(
                    &tx,
                    run_id,
                    &PlanLinkRecord::from_input(PlanLinkInput {
                        source_type: "candidate",
                        source_id: parent_id,
                        target_type: "candidate",
                        target_id: &candidate_id,
                        relation: "candidate_lineage",
                        status: "active",
                        confidence: 1.0,
                        metadata: Map::new(),
                    }),
                )?;
                upsert_plan_link_tx(
                    &tx,
                    run_id,
                    &PlanLinkRecord::from_input(PlanLinkInput {
                        source_type: "candidate_delta",
                        source_id: &delta.candidate_delta_id,
                        target_type: "candidate",
                        target_id: &candidate_id,
                        relation: "candidate_delta",
                        status: "active",
                        confidence: 1.0,
                        metadata: Map::new(),
                    }),
                )?;
                Some(snapshot)
            } else {
                None
            };
            let acceptance = AcceptanceDecisionRecord::from_input(AcceptanceDecisionInput {
                candidate_id: &candidate_id,
                parent_candidate_id: parent_id.clone(),
                candidate_status: &status,
                candidate_minibatch_reward: candidate
                    .get("minibatch_reward")
                    .and_then(Value::as_f64),
                parent_minibatch_reward: parent_snapshot
                    .as_ref()
                    .and_then(|snapshot| snapshot.minibatch_reward),
                candidate_train_reward: candidate.get("train_reward").and_then(Value::as_f64),
                parent_train_reward: parent_snapshot
                    .as_ref()
                    .and_then(|snapshot| snapshot.train_reward),
                heldout_reward: candidate.get("heldout_reward").and_then(Value::as_f64),
                score: candidate
                    .get("acceptance_score")
                    .filter(|value| !value.is_null())
                    .cloned(),
                metadata: candidate
                    .get("acceptance_metadata")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default(),
            });
            upsert_acceptance_decision_tx(&tx, run_id, &acceptance)?;
            upsert_plan_link_tx(
                &tx,
                run_id,
                &PlanLinkRecord::from_input(PlanLinkInput {
                    source_type: "acceptance_decision",
                    source_id: &acceptance.acceptance_decision_id,
                    target_type: "candidate",
                    target_id: &candidate_id,
                    relation: "acceptance_decision",
                    status: "active",
                    confidence: 1.0,
                    metadata: Map::new(),
                }),
            )?;
            for frame in candidate
                .get("sensor_frames")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
            {
                let frame: SensorFrame = serde_json::from_value(frame)?;
                upsert_sensor_frame_tx(&tx, run_id, &frame)?;
                upsert_plan_link_tx(
                    &tx,
                    run_id,
                    &PlanLinkRecord::from_input(PlanLinkInput {
                        source_type: "candidate",
                        source_id: &candidate_id,
                        target_type: "sensor_frame",
                        target_id: &frame.sensor_frame_id,
                        relation: "rollout_observation",
                        status: "active",
                        confidence: 1.0,
                        metadata: Map::new(),
                    }),
                )?;
                upsert_rollout_job_tx(&tx, run_id, &frame)?;
                let rollout_records = SensorRolloutRecords::from_sensor_frame(&frame);
                upsert_rollout_record_tx(&tx, run_id, &rollout_records.rollout)?;
                for event in &rollout_records.events {
                    upsert_rollout_event_tx(&tx, run_id, event)?;
                }
                let score_records = SensorScoreRecords::from_sensor_frame(&frame);
                for objective in &score_records.objectives {
                    upsert_objective_tx(&tx, run_id, objective)?;
                }
                for score in &score_records.scores {
                    upsert_score_tx(&tx, run_id, score)?;
                }
                let derived = SensorDerivedRecords::from_sensor_frame(&frame);
                upsert_trace_annotation_tx(&tx, run_id, &derived.trace_annotation)?;
                for evidence_frame in &derived.evidence_frames {
                    upsert_evidence_frame_tx(&tx, run_id, evidence_frame)?;
                }
                upsert_verifier_job_tx(&tx, run_id, &derived.verifier_job)?;
                upsert_subagent_invocation_tx(&tx, run_id, &derived.subagent_invocation)?;
                let annotation_job_id =
                    format!("annotation:{}", &derived.trace_annotation.annotation_id);
                upsert_optimizer_job_tx(
                    &tx,
                    run_id,
                    &OptimizerJobPersist {
                        job_id: &annotation_job_id,
                        kind: OptimizerJobKind::Annotation,
                        status: OptimizerJobStatus::Completed,
                        candidate_id: Some(&frame.candidate_id),
                        sensor_frame_id: Some(&frame.sensor_frame_id),
                        failure: None,
                        payload: json!({
                        "annotation_id": &derived.trace_annotation.annotation_id,
                        "status": &derived.trace_annotation.status,
                        }),
                    },
                )?;
                upsert_optimizer_job_tx(
                    &tx,
                    run_id,
                    &OptimizerJobPersist {
                        job_id: &derived.verifier_job.verifier_job_id,
                        kind: OptimizerJobKind::Verification,
                        status: OptimizerJobStatus::Completed,
                        candidate_id: Some(&frame.candidate_id),
                        sensor_frame_id: Some(&frame.sensor_frame_id),
                        failure: derived.verifier_job.failure.as_ref(),
                        payload: serde_json::to_value(&derived.verifier_job)?,
                    },
                )?;
                let subagent_job_id =
                    format!("subagent:{}", &derived.subagent_invocation.invocation_id);
                upsert_optimizer_job_tx(
                    &tx,
                    run_id,
                    &OptimizerJobPersist {
                        job_id: &subagent_job_id,
                        kind: OptimizerJobKind::Subagent,
                        status: OptimizerJobStatus::Completed,
                        candidate_id: Some(&frame.candidate_id),
                        sensor_frame_id: Some(&frame.sensor_frame_id),
                        failure: derived.subagent_invocation.failure.as_ref(),
                        payload: serde_json::to_value(&derived.subagent_invocation)?,
                    },
                )?;
            }
        }
        let (objective_set, score_vectors) = rebuild_score_vectors_tx(&tx, run_id)?;
        if let Some(objective_set) = objective_set.as_ref() {
            rebuild_pareto_comparisons_tx(&tx, run_id, objective_set, &score_vectors)?;
        }
        rebuild_frontier_cells_tx(&tx, run_id)?;
        tx.commit()?;
        Ok(())
    }

    pub fn record_artifact_refs(
        &mut self,
        run_id: &str,
        artifact_refs: &[ArtifactRef],
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        for artifact in artifact_refs {
            tx.execute(
                r#"
                INSERT INTO artifact_refs(run_id, path, kind, sha256, bytes, retention)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                ON CONFLICT(run_id, path) DO UPDATE SET
                    kind = excluded.kind,
                    sha256 = excluded.sha256,
                    bytes = excluded.bytes,
                    retention = excluded.retention
                "#,
                params![
                    run_id,
                    artifact.path,
                    artifact.kind,
                    artifact.sha256,
                    artifact.bytes as i64,
                    artifact.retention,
                ],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn record_evidence_frames_and_plan_links(
        &mut self,
        run_id: &str,
        evidence_frames: &[EvidenceFrame],
        plan_links: &[PlanLinkRecord],
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        for frame in evidence_frames {
            upsert_evidence_frame_tx(&tx, run_id, frame)?;
        }
        for link in plan_links {
            upsert_plan_link_tx(&tx, run_id, link)?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn record_manifest(
        &self,
        run_id: &str,
        manifest_path: &Path,
        best_candidate_id: &str,
        cost_usd: f64,
        usage: &Value,
        manifest: &Value,
    ) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO manifests(
                run_id, manifest_path, best_candidate_id, cost_usd,
                usage_json, manifest_json, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, datetime('now'))
            ON CONFLICT(run_id) DO UPDATE SET
                manifest_path = excluded.manifest_path,
                best_candidate_id = excluded.best_candidate_id,
                cost_usd = excluded.cost_usd,
                usage_json = excluded.usage_json,
                manifest_json = excluded.manifest_json,
                updated_at = datetime('now')
            "#,
            params![
                run_id,
                manifest_path.display().to_string(),
                best_candidate_id,
                cost_usd,
                stable_json(usage),
                stable_json(manifest),
            ],
        )?;
        Ok(())
    }

    pub fn record_cache_profile(
        &mut self,
        run_id: &str,
        profile: &CacheProfileRecord,
        accesses: &[CacheAccessRecord],
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            r#"
            INSERT INTO cache_profiles(
                run_id, cache_profile_id, mode, path, entries, hits, misses,
                writes, total_accesses, profile_json, profile_record_json,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, datetime('now'))
            ON CONFLICT(run_id, cache_profile_id) DO UPDATE SET
                mode = excluded.mode,
                path = excluded.path,
                entries = excluded.entries,
                hits = excluded.hits,
                misses = excluded.misses,
                writes = excluded.writes,
                total_accesses = excluded.total_accesses,
                profile_json = excluded.profile_json,
                profile_record_json = excluded.profile_record_json,
                updated_at = datetime('now')
            "#,
            params![
                run_id,
                profile.cache_profile_id,
                profile.mode,
                profile.path,
                profile.entries as i64,
                profile.hits as i64,
                profile.misses as i64,
                profile.writes as i64,
                profile.total_accesses as i64,
                stable_json(&serde_json::to_value(&profile.profile)?),
                stable_json(&serde_json::to_value(profile)?),
            ],
        )?;
        for access in accesses {
            upsert_cache_access_tx(&tx, run_id, access)?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn record_objective_set(&self, run_id: &str, record: &ObjectiveSetRecord) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO objective_sets(
                run_id, objective_set_id, objective_set_hash,
                selection_objective, frontier_type, objectives_json,
                metadata_json, objective_set_json, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, datetime('now'))
            ON CONFLICT(run_id, objective_set_id) DO UPDATE SET
                objective_set_hash = excluded.objective_set_hash,
                selection_objective = excluded.selection_objective,
                frontier_type = excluded.frontier_type,
                objectives_json = excluded.objectives_json,
                metadata_json = excluded.metadata_json,
                objective_set_json = excluded.objective_set_json,
                updated_at = datetime('now')
            "#,
            params![
                run_id,
                &record.objective_set_id,
                &record.objective_set_hash,
                &record.selection_objective,
                &record.frontier_type,
                stable_json(&serde_json::to_value(&record.objectives)?),
                stable_json(&serde_json::to_value(&record.metadata)?),
                stable_json(&serde_json::to_value(record)?),
            ],
        )?;
        Ok(())
    }

    pub fn record_score_vector(&mut self, run_id: &str, record: &ScoreVectorRecord) -> Result<()> {
        let tx = self.conn.transaction()?;
        upsert_score_vector_tx(&tx, run_id, record)?;
        tx.commit()?;
        Ok(())
    }

    pub fn record_pareto_comparison(
        &mut self,
        run_id: &str,
        record: &ParetoComparisonRecord,
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        upsert_pareto_comparison_tx(&tx, run_id, record)?;
        tx.commit()?;
        Ok(())
    }

    pub fn record_materialization(
        &self,
        run_id: &str,
        record: &MaterializationRecord,
    ) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO materializations(
                run_id, materialization_id, candidate_id, example_id, seed,
                split, evaluation_stage, task_id, evaluator_id, algorithm_id,
                materializer_id, lever_version, sensor_version,
                objective_set_hash, candidate_hash, example_hash, request_hash,
                cache_key, platform_cache_key, status, request_json,
                candidate_payload_json, dataset_row_json, metadata_json,
                record_json, created_at, updated_at
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13,
                ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24,
                ?25, datetime('now'), datetime('now')
            )
            ON CONFLICT(run_id, materialization_id) DO UPDATE SET
                candidate_id = excluded.candidate_id,
                example_id = excluded.example_id,
                seed = excluded.seed,
                split = excluded.split,
                evaluation_stage = excluded.evaluation_stage,
                task_id = excluded.task_id,
                evaluator_id = excluded.evaluator_id,
                algorithm_id = excluded.algorithm_id,
                materializer_id = excluded.materializer_id,
                lever_version = excluded.lever_version,
                sensor_version = excluded.sensor_version,
                objective_set_hash = excluded.objective_set_hash,
                candidate_hash = excluded.candidate_hash,
                example_hash = excluded.example_hash,
                request_hash = excluded.request_hash,
                cache_key = excluded.cache_key,
                platform_cache_key = excluded.platform_cache_key,
                status = excluded.status,
                request_json = excluded.request_json,
                candidate_payload_json = excluded.candidate_payload_json,
                dataset_row_json = excluded.dataset_row_json,
                metadata_json = excluded.metadata_json,
                record_json = excluded.record_json,
                updated_at = datetime('now')
            "#,
            params![
                run_id,
                &record.materialization_id,
                &record.candidate_id,
                &record.example_id,
                record.seed,
                &record.split,
                &record.evaluation_stage,
                &record.task_id,
                &record.evaluator_id,
                &record.algorithm_id,
                &record.materializer_id,
                &record.lever_version,
                &record.sensor_version,
                &record.objective_set_hash,
                &record.candidate_hash,
                &record.example_hash,
                &record.request_hash,
                &record.cache_key,
                record.platform_cache_key.as_deref(),
                &record.status,
                stable_json(&record.request),
                stable_json(&record.candidate_payload),
                stable_json(&record.dataset_row),
                stable_json(&Value::Object(record.metadata.clone())),
                materialization_record_json(record),
            ],
        )?;
        Ok(())
    }

    pub fn record_evaluation_cache(
        &self,
        run_id: &str,
        record: &EvaluationCacheRecord,
    ) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO evaluation_cache(
                run_id, cache_key, cache_schema_version, cache_profile,
                cache_key_fields_json, candidate_hash, example_hash,
                request_hash, example_id, evaluator_id, algorithm_id,
                materializer_id, lever_version, sensor_version,
                objective_set_hash, source_rollout_id, reward,
                objective_scores_json, actionable_side_info_json, usage_json,
                trace_ref, status, cache_hit, platform_cache_key,
                rollout_payload_json, metadata_json, record_json,
                created_at, updated_at
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13,
                ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24,
                ?25, ?26, ?27, datetime('now'), datetime('now')
            )
            ON CONFLICT(run_id, cache_key) DO UPDATE SET
                cache_schema_version = excluded.cache_schema_version,
                cache_profile = excluded.cache_profile,
                cache_key_fields_json = excluded.cache_key_fields_json,
                candidate_hash = excluded.candidate_hash,
                example_hash = excluded.example_hash,
                request_hash = excluded.request_hash,
                example_id = excluded.example_id,
                evaluator_id = excluded.evaluator_id,
                algorithm_id = excluded.algorithm_id,
                materializer_id = excluded.materializer_id,
                lever_version = excluded.lever_version,
                sensor_version = excluded.sensor_version,
                objective_set_hash = excluded.objective_set_hash,
                source_rollout_id = excluded.source_rollout_id,
                reward = excluded.reward,
                objective_scores_json = excluded.objective_scores_json,
                actionable_side_info_json = excluded.actionable_side_info_json,
                usage_json = excluded.usage_json,
                trace_ref = excluded.trace_ref,
                status = excluded.status,
                cache_hit = excluded.cache_hit,
                platform_cache_key = excluded.platform_cache_key,
                rollout_payload_json = excluded.rollout_payload_json,
                metadata_json = excluded.metadata_json,
                record_json = excluded.record_json,
                updated_at = datetime('now')
            "#,
            params![
                run_id,
                &record.cache_key,
                &record.schema_version,
                &record.cache_profile,
                stable_json(&record.cache_key_fields),
                &record.candidate_hash,
                &record.example_hash,
                &record.request_hash,
                &record.example_id,
                &record.evaluator_id,
                &record.algorithm_id,
                &record.materializer_id,
                &record.lever_version,
                &record.sensor_version,
                &record.objective_set_hash,
                record.source_rollout_id.as_deref(),
                record.reward,
                stable_json(&record.objective_scores),
                stable_json(&record.actionable_side_info),
                stable_json(&record.usage),
                record.trace_ref.as_deref(),
                &record.status,
                if record.cache_hit { 1 } else { 0 },
                record.platform_cache_key.as_deref(),
                stable_json(&record.rollout_payload),
                stable_json(&Value::Object(record.metadata.clone())),
                record_json(record),
            ],
        )?;
        Ok(())
    }

    pub fn record_event_stream(
        &mut self,
        run_id: &str,
        events: &[EventStreamRecord],
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        for event in events {
            upsert_event_stream_event_tx(&tx, run_id, event)?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn record_usage_ledger(
        &mut self,
        run_id: &str,
        records: &[UsageLedgerRecord],
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        for record in records {
            upsert_usage_ledger_tx(&tx, run_id, record)?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn record_stopper_states(
        &mut self,
        run_id: &str,
        records: &[StopperStateRecord],
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        for record in records {
            upsert_stopper_state_tx(&tx, run_id, record)?;
        }
        tx.commit()?;
        Ok(())
    }

    pub fn record_checkpoint(&mut self, run_id: &str, record: &CheckpointRecord) -> Result<()> {
        let tx = self.conn.transaction()?;
        upsert_checkpoint_tx(&tx, run_id, record)?;
        tx.commit()?;
        Ok(())
    }

    pub fn latest_checkpoint(
        &self,
        run_id: &str,
        checkpoint_kind: &str,
    ) -> Result<Option<CheckpointRecord>> {
        self.conn
            .query_row(
                r#"
                SELECT checkpoint_json
                FROM checkpoints
                WHERE run_id = ?1 AND checkpoint_kind = ?2
                ORDER BY sequence_number DESC, created_at DESC
                LIMIT 1
                "#,
                params![run_id, checkpoint_kind],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .map(|raw| serde_json::from_str(&raw).map_err(OptimizerError::from))
            .transpose()
    }

    pub fn checkpoint_history(
        &self,
        run_id: &str,
        checkpoint_kind: Option<&str>,
    ) -> Result<Vec<CheckpointRecord>> {
        let mut records = Vec::new();
        if let Some(kind) = checkpoint_kind {
            let mut stmt = self.conn.prepare(
                r#"
                SELECT checkpoint_json
                FROM checkpoints
                WHERE run_id = ?1 AND checkpoint_kind = ?2
                ORDER BY sequence_number ASC, created_at ASC
                "#,
            )?;
            let mut rows = stmt.query(params![run_id, kind])?;
            while let Some(row) = rows.next()? {
                let raw: String = row.get(0)?;
                records.push(serde_json::from_str(&raw)?);
            }
            return Ok(records);
        }
        let mut stmt = self.conn.prepare(
            r#"
            SELECT checkpoint_json
            FROM checkpoints
            WHERE run_id = ?1
            ORDER BY sequence_number ASC, created_at ASC
            "#,
        )?;
        let mut rows = stmt.query(params![run_id])?;
        while let Some(row) = rows.next()? {
            let raw: String = row.get(0)?;
            records.push(serde_json::from_str(&raw)?);
        }
        Ok(records)
    }

    pub fn event_stream_history(&self, run_id: &str) -> Result<Vec<EventStreamRecord>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT event_record_json
            FROM event_stream_events
            WHERE run_id = ?1
            ORDER BY sequence_number ASC, event_id ASC
            "#,
        )?;
        let mut rows = stmt.query(params![run_id])?;
        let mut records = Vec::new();
        while let Some(row) = rows.next()? {
            let raw: String = row.get(0)?;
            records.push(serde_json::from_str(&raw)?);
        }
        Ok(records)
    }

    pub fn current_optimizer_state(&self, run_id: &str) -> Result<Option<WorkspaceRunStatus>> {
        Ok(self
            .status()?
            .runs
            .into_iter()
            .find(|run| run.run_id == run_id))
    }

    pub fn record_run_finished(
        &self,
        run_id: &str,
        best_candidate_id: &str,
        cost_usd: f64,
        usage: &Value,
    ) -> Result<()> {
        self.conn.execute(
            r#"
            UPDATE optimization_runs
            SET state = 'completed',
                best_candidate_id = ?1,
                cost_usd = ?2,
                usage_json = ?3,
                completed_at = datetime('now'),
                updated_at = datetime('now')
            WHERE run_id = ?4
            "#,
            params![best_candidate_id, cost_usd, stable_json(usage), run_id],
        )?;
        Ok(())
    }

    pub fn record_run_failed(
        &self,
        run_id: &str,
        best_candidate_id: Option<&str>,
        cost_usd: f64,
        usage: &Value,
    ) -> Result<()> {
        self.conn.execute(
            r#"
            UPDATE optimization_runs
            SET state = 'failed',
                best_candidate_id = COALESCE(?1, best_candidate_id),
                cost_usd = ?2,
                usage_json = ?3,
                completed_at = COALESCE(completed_at, datetime('now')),
                updated_at = datetime('now')
            WHERE run_id = ?4
            "#,
            params![best_candidate_id, cost_usd, stable_json(usage), run_id],
        )?;
        Ok(())
    }

    pub fn record_run_cancelled_result(
        &self,
        run_id: &str,
        best_candidate_id: Option<&str>,
        cost_usd: f64,
        usage: &Value,
    ) -> Result<()> {
        self.conn.execute(
            r#"
            UPDATE optimization_runs
            SET state = 'cancelled',
                best_candidate_id = COALESCE(?1, best_candidate_id),
                cost_usd = ?2,
                usage_json = ?3,
                completed_at = COALESCE(completed_at, datetime('now')),
                updated_at = datetime('now')
            WHERE run_id = ?4
            "#,
            params![best_candidate_id, cost_usd, stable_json(usage), run_id],
        )?;
        Ok(())
    }

    pub fn record_run_cancelled(&self, run_id: &str) -> Result<()> {
        let from_state = self
            .conn
            .query_row(
                "SELECT state FROM optimization_runs WHERE run_id = ?1",
                params![run_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .unwrap_or_else(|| "unknown".to_string());
        let sequence_number = self
            .conn
            .query_row(
                "SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM optimizer_state_history WHERE run_id = ?1",
                params![run_id],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(1);
        self.conn.execute(
            r#"
            INSERT INTO optimizer_state_history(
                run_id, sequence_number, from_state, to_state, trigger,
                message, transition_at, details_json
            ) VALUES (?1, ?2, ?3, 'cancelled', 'cancel_requested', ?4, ?5, ?6)
            ON CONFLICT(run_id, sequence_number) DO UPDATE SET
                from_state = excluded.from_state,
                to_state = excluded.to_state,
                trigger = excluded.trigger,
                message = excluded.message,
                transition_at = excluded.transition_at,
                details_json = excluded.details_json
            "#,
            params![
                run_id,
                sequence_number,
                from_state,
                "GEPA run cancelled",
                now_rfc3339(),
                stable_json(&json!({"source": "service_request_cancelled"})),
            ],
        )?;
        self.conn.execute(
            r#"
            UPDATE optimization_runs
            SET state = 'cancelled',
                completed_at = COALESCE(completed_at, datetime('now')),
                updated_at = datetime('now')
            WHERE run_id = ?1
            "#,
            params![run_id],
        )?;
        Ok(())
    }

    fn initialize(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            PRAGMA foreign_keys = ON;
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS workspace_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            INSERT INTO workspace_metadata(key, value)
            VALUES ('schema_version', 'synth_optimizers.workspace.v1')
            ON CONFLICT(key) DO UPDATE SET value = excluded.value;

            CREATE TABLE IF NOT EXISTS optimization_runs (
                run_id TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                config_json TEXT NOT NULL,
                cache_mode TEXT NOT NULL,
                cache_namespace TEXT NOT NULL,
                output_dir TEXT NOT NULL,
                run_dir TEXT NOT NULL,
                manifest_path TEXT NOT NULL,
                best_candidate_id TEXT,
                cost_usd REAL,
                usage_json TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS run_requests (
                request_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                status TEXT NOT NULL,
                config_path TEXT NOT NULL,
                config_json TEXT NOT NULL,
                container_url TEXT NOT NULL,
                cache_mode TEXT NOT NULL,
                cache_namespace TEXT NOT NULL,
                output_dir TEXT NOT NULL,
                run_dir TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                submitted_at TEXT NOT NULL,
                leased_at TEXT,
                lease_expires_at TEXT,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL,
                lease_id TEXT,
                worker_id TEXT,
                run_workspace_db_path TEXT,
                result_manifest_path TEXT,
                best_candidate_id TEXT,
                cost_usd REAL,
                usage_json TEXT,
                result_json TEXT,
                error_json TEXT
            );

            CREATE TABLE IF NOT EXISTS operations (
                operation_id TEXT PRIMARY KEY,
                operation_type TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                idempotency_key TEXT,
                status TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                operation_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS resource_leases (
                resource_lease_id TEXT PRIMARY KEY,
                lease_id TEXT NOT NULL,
                resource_kind TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                status TEXT NOT NULL,
                lease_expires_at TEXT,
                metadata_json TEXT NOT NULL,
                lease_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS optimizer_state_history (
                run_id TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                from_state TEXT NOT NULL,
                to_state TEXT NOT NULL,
                trigger TEXT NOT NULL,
                message TEXT NOT NULL,
                transition_at TEXT NOT NULL,
                details_json TEXT NOT NULL,
                PRIMARY KEY(run_id, sequence_number),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS rendered_optimizer_states (
                run_id TEXT NOT NULL,
                rendered_state_id TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                run_phase TEXT NOT NULL,
                generation_phase TEXT,
                candidate_phase TEXT,
                block_status TEXT NOT NULL,
                terminal_status TEXT,
                best_candidate_id TEXT,
                frontier_size INTEGER NOT NULL,
                active_effect_count INTEGER NOT NULL,
                active_job_count INTEGER NOT NULL,
                queue_counts_json TEXT NOT NULL,
                budget_status_json TEXT NOT NULL,
                evidence_status_json TEXT NOT NULL,
                details_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                rendered_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, rendered_state_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS resolved_run_configs (
                run_id TEXT NOT NULL,
                resolved_config_id TEXT NOT NULL,
                algorithm_id TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                cache_mode TEXT NOT NULL,
                cache_namespace TEXT NOT NULL,
                output_dir TEXT NOT NULL,
                config_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, resolved_config_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS container_contract_snapshots (
                run_id TEXT NOT NULL,
                contract_snapshot_id TEXT NOT NULL,
                container_url TEXT NOT NULL,
                contract_kind TEXT NOT NULL,
                contract_version TEXT NOT NULL,
                validation_status TEXT NOT NULL,
                capability_hash TEXT NOT NULL,
                metadata_response_json TEXT NOT NULL,
                health_response_json TEXT,
                metadata_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, contract_snapshot_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS prompt_program_snapshots (
                run_id TEXT NOT NULL,
                program_snapshot_id TEXT NOT NULL,
                program_id TEXT NOT NULL,
                program_hash TEXT NOT NULL,
                target_modules_json TEXT NOT NULL,
                mutable_field_ids_json TEXT NOT NULL,
                validation_status TEXT NOT NULL,
                program_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, program_snapshot_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS dataset_snapshots (
                run_id TEXT NOT NULL,
                dataset_snapshot_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                split TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                seed_count INTEGER NOT NULL,
                seeds_json TEXT NOT NULL,
                filters_json TEXT NOT NULL,
                rows_hash TEXT NOT NULL,
                rows_json TEXT NOT NULL,
                dataset_metadata_json TEXT NOT NULL,
                rows_metadata_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, dataset_snapshot_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS run_limits (
                run_id TEXT NOT NULL,
                run_limits_id TEXT NOT NULL,
                max_total_rollouts INTEGER,
                max_cost_usd REAL,
                max_time_seconds INTEGER,
                max_prompt_tokens INTEGER,
                max_completion_tokens INTEGER,
                max_total_tokens INTEGER,
                hard_limit INTEGER NOT NULL,
                stop_policy TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, run_limits_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS runtime_effects (
                run_id TEXT NOT NULL,
                runtime_effect_id TEXT NOT NULL,
                effect_kind TEXT NOT NULL,
                lane TEXT NOT NULL,
                status TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                cache_key TEXT,
                job_id TEXT,
                budget_reservation_id TEXT,
                attempt INTEGER NOT NULL,
                failure_class TEXT,
                payload_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                planned_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                terminal_at TEXT,
                PRIMARY KEY(run_id, runtime_effect_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS runtime_effect_admissions (
                run_id TEXT NOT NULL,
                admission_id TEXT NOT NULL,
                runtime_effect_id TEXT NOT NULL,
                effect_kind TEXT NOT NULL,
                lane TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                status TEXT NOT NULL,
                rejection_reason TEXT,
                max_cost_usd REAL,
                max_prompt_tokens INTEGER,
                max_completion_tokens INTEGER,
                max_total_tokens INTEGER,
                max_rollouts INTEGER,
                max_wall_seconds INTEGER,
                ledger_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                PRIMARY KEY(run_id, admission_id),
                FOREIGN KEY(run_id, runtime_effect_id) REFERENCES runtime_effects(run_id, runtime_effect_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS budget_reservations (
                run_id TEXT NOT NULL,
                budget_reservation_id TEXT NOT NULL,
                runtime_effect_id TEXT NOT NULL,
                status TEXT NOT NULL,
                max_cost_usd REAL,
                max_prompt_tokens INTEGER,
                max_completion_tokens INTEGER,
                max_total_tokens INTEGER,
                max_rollouts INTEGER,
                max_wall_seconds INTEGER,
                metadata_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                reserved_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, budget_reservation_id),
                FOREIGN KEY(run_id, runtime_effect_id) REFERENCES runtime_effects(run_id, runtime_effect_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS budget_commits (
                run_id TEXT NOT NULL,
                budget_commit_id TEXT NOT NULL,
                runtime_effect_id TEXT NOT NULL,
                budget_reservation_id TEXT NOT NULL,
                cost_usd REAL NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                rollout_count INTEGER NOT NULL,
                wall_seconds INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                committed_at TEXT NOT NULL,
                PRIMARY KEY(run_id, budget_commit_id),
                FOREIGN KEY(run_id, runtime_effect_id) REFERENCES runtime_effects(run_id, runtime_effect_id) ON DELETE CASCADE,
                FOREIGN KEY(run_id, budget_reservation_id) REFERENCES budget_reservations(run_id, budget_reservation_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS budget_releases (
                run_id TEXT NOT NULL,
                budget_release_id TEXT NOT NULL,
                runtime_effect_id TEXT NOT NULL,
                budget_reservation_id TEXT NOT NULL,
                release_reason TEXT NOT NULL,
                released_cost_usd REAL NOT NULL,
                released_prompt_tokens INTEGER NOT NULL,
                released_completion_tokens INTEGER NOT NULL,
                released_total_tokens INTEGER NOT NULL,
                released_rollouts INTEGER NOT NULL,
                released_wall_seconds INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                released_at TEXT NOT NULL,
                PRIMARY KEY(run_id, budget_release_id),
                FOREIGN KEY(run_id, runtime_effect_id) REFERENCES runtime_effects(run_id, runtime_effect_id) ON DELETE CASCADE,
                FOREIGN KEY(run_id, budget_reservation_id) REFERENCES budget_reservations(run_id, budget_reservation_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS candidates (
                run_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                parent_id TEXT,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                lever_bundle_json TEXT NOT NULL,
                minibatch_reward REAL,
                train_reward REAL,
                heldout_reward REAL,
                record_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, candidate_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS candidate_payloads (
                run_id TEXT NOT NULL,
                candidate_payload_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                parent_id TEXT,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                lever_bundle_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                payload_record_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, candidate_payload_id),
                FOREIGN KEY(run_id, candidate_id) REFERENCES candidates(run_id, candidate_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS candidate_deltas (
                run_id TEXT NOT NULL,
                candidate_delta_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                parent_candidate_id TEXT NOT NULL,
                operation_kind TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                target_levers_json TEXT NOT NULL,
                changed_fields_json TEXT NOT NULL,
                before_json TEXT NOT NULL,
                after_json TEXT NOT NULL,
                rationale TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                delta_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, candidate_delta_id),
                FOREIGN KEY(run_id, candidate_id) REFERENCES candidates(run_id, candidate_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS acceptance_decisions (
                run_id TEXT NOT NULL,
                acceptance_decision_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                parent_candidate_id TEXT,
                decision TEXT NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                candidate_minibatch_reward REAL,
                parent_minibatch_reward REAL,
                candidate_train_reward REAL,
                parent_train_reward REAL,
                heldout_reward REAL,
                score_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                decision_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, acceptance_decision_id),
                FOREIGN KEY(run_id, candidate_id) REFERENCES candidates(run_id, candidate_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS frontier_cells (
                run_id TEXT NOT NULL,
                frontier_cell_id TEXT NOT NULL,
                frontier_name TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                parent_candidate_id TEXT,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                split TEXT NOT NULL,
                objective TEXT NOT NULL,
                rank INTEGER NOT NULL,
                score REAL NOT NULL,
                score_vector_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                cell_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, frontier_cell_id),
                FOREIGN KEY(run_id, candidate_id) REFERENCES candidates(run_id, candidate_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS plan_links (
                run_id TEXT NOT NULL,
                plan_link_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                metadata_json TEXT NOT NULL,
                link_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, plan_link_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS cache_profiles (
                run_id TEXT NOT NULL,
                cache_profile_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                path TEXT NOT NULL,
                entries INTEGER NOT NULL,
                hits INTEGER NOT NULL,
                misses INTEGER NOT NULL,
                writes INTEGER NOT NULL,
                total_accesses INTEGER NOT NULL,
                profile_json TEXT NOT NULL,
                profile_record_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, cache_profile_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS cache_accesses (
                run_id TEXT NOT NULL,
                cache_access_id TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                mode TEXT NOT NULL,
                namespace TEXT NOT NULL,
                boundary TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                request_hash TEXT,
                response_hash TEXT,
                metadata_json TEXT NOT NULL,
                access_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, cache_access_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS materializations (
                run_id TEXT NOT NULL,
                materialization_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                example_id TEXT NOT NULL,
                seed INTEGER NOT NULL,
                split TEXT NOT NULL,
                evaluation_stage TEXT NOT NULL,
                task_id TEXT NOT NULL,
                evaluator_id TEXT NOT NULL,
                algorithm_id TEXT NOT NULL,
                materializer_id TEXT NOT NULL,
                lever_version TEXT NOT NULL,
                sensor_version TEXT NOT NULL,
                objective_set_hash TEXT NOT NULL,
                candidate_hash TEXT NOT NULL,
                example_hash TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                platform_cache_key TEXT,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                candidate_payload_json TEXT NOT NULL,
                dataset_row_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, materialization_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS evaluation_cache (
                run_id TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                cache_schema_version TEXT NOT NULL,
                cache_profile TEXT NOT NULL,
                cache_key_fields_json TEXT NOT NULL,
                candidate_hash TEXT NOT NULL,
                example_hash TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                example_id TEXT NOT NULL,
                evaluator_id TEXT NOT NULL,
                algorithm_id TEXT NOT NULL,
                materializer_id TEXT NOT NULL,
                lever_version TEXT NOT NULL,
                sensor_version TEXT NOT NULL,
                objective_set_hash TEXT NOT NULL,
                source_rollout_id TEXT,
                reward REAL NOT NULL,
                objective_scores_json TEXT NOT NULL,
                actionable_side_info_json TEXT NOT NULL,
                usage_json TEXT NOT NULL,
                trace_ref TEXT,
                status TEXT NOT NULL,
                cache_hit INTEGER NOT NULL,
                platform_cache_key TEXT,
                rollout_payload_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, cache_key),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS event_stream_events (
                run_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                fields_json TEXT NOT NULL,
                event_json TEXT NOT NULL,
                event_record_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, event_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS usage_ledger (
                run_id TEXT NOT NULL,
                usage_ledger_id TEXT NOT NULL,
                boundary TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                candidate_id TEXT,
                evaluation_stage TEXT,
                model TEXT,
                provider TEXT,
                call_count INTEGER NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                usage_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                ledger_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, usage_ledger_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS stopper_states (
                run_id TEXT NOT NULL,
                stopper_state_id TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                reason TEXT,
                generation INTEGER,
                candidate_id TEXT,
                evaluation_stage TEXT,
                rollout_count INTEGER NOT NULL,
                max_total_rollouts INTEGER NOT NULL,
                remaining_rollouts INTEGER,
                cost_usd REAL NOT NULL,
                max_cost_usd REAL NOT NULL,
                cost_budget_enabled INTEGER NOT NULL,
                budget_exhausted INTEGER NOT NULL,
                checked_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, stopper_state_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                run_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                checkpoint_kind TEXT NOT NULL,
                status TEXT NOT NULL,
                run_state TEXT NOT NULL,
                reason TEXT,
                generation INTEGER,
                candidate_id TEXT,
                evaluation_stage TEXT,
                best_candidate_id TEXT,
                candidate_count INTEGER NOT NULL,
                frontier_count INTEGER NOT NULL,
                rollout_count INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                usage_json TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                checkpoint_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, checkpoint_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS optimizer_jobs (
                run_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                candidate_id TEXT,
                sensor_frame_id TEXT,
                attempt INTEGER NOT NULL DEFAULT 1,
                lease_id TEXT,
                worker_id TEXT,
                leased_at TEXT,
                lease_expires_at TEXT,
                heartbeat_at TEXT,
                next_retry_at TEXT,
                retry_policy_json TEXT NOT NULL,
                failure_json TEXT,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, job_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS rollout_jobs (
                run_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                sensor_frame_id TEXT NOT NULL,
                example_id TEXT NOT NULL,
                seed INTEGER NOT NULL,
                split TEXT NOT NULL,
                evaluation_stage TEXT NOT NULL,
                status TEXT NOT NULL,
                reward REAL NOT NULL,
                failure_json TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, job_id),
                FOREIGN KEY(run_id, candidate_id) REFERENCES candidates(run_id, candidate_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS sensor_frames (
                run_id TEXT NOT NULL,
                sensor_frame_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                rollout_id TEXT,
                example_id TEXT NOT NULL,
                seed INTEGER NOT NULL,
                split TEXT NOT NULL,
                evaluation_stage TEXT NOT NULL,
                reward REAL NOT NULL,
                status TEXT NOT NULL,
                trace_digest_json TEXT,
                usage_json TEXT NOT NULL,
                failure_json TEXT,
                frame_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, sensor_frame_id),
                FOREIGN KEY(run_id, candidate_id) REFERENCES candidates(run_id, candidate_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS rollouts (
                run_id TEXT NOT NULL,
                rollout_record_id TEXT NOT NULL,
                rollout_id TEXT,
                candidate_id TEXT NOT NULL,
                sensor_frame_id TEXT NOT NULL,
                example_id TEXT NOT NULL,
                seed INTEGER NOT NULL,
                split TEXT NOT NULL,
                evaluation_stage TEXT NOT NULL,
                status TEXT NOT NULL,
                reward REAL NOT NULL,
                trace_sha256 TEXT,
                event_count INTEGER NOT NULL,
                usage_json TEXT NOT NULL,
                failure_json TEXT,
                metadata_json TEXT NOT NULL,
                rollout_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, rollout_record_id),
                FOREIGN KEY(run_id, sensor_frame_id) REFERENCES sensor_frames(run_id, sensor_frame_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS rollout_events (
                run_id TEXT NOT NULL,
                rollout_event_id TEXT NOT NULL,
                rollout_record_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                sensor_frame_id TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                trace_ref TEXT,
                event_json TEXT NOT NULL,
                event_record_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, rollout_event_id),
                FOREIGN KEY(run_id, rollout_record_id) REFERENCES rollouts(run_id, rollout_record_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS objective_sets (
                run_id TEXT NOT NULL,
                objective_set_id TEXT NOT NULL,
                objective_set_hash TEXT NOT NULL,
                selection_objective TEXT NOT NULL,
                frontier_type TEXT NOT NULL,
                objectives_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                objective_set_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, objective_set_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS objectives (
                run_id TEXT NOT NULL,
                objective_id TEXT NOT NULL,
                name TEXT NOT NULL,
                direction TEXT NOT NULL,
                source TEXT NOT NULL,
                aggregation TEXT NOT NULL,
                split_policy TEXT NOT NULL,
                objective_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, objective_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS scores (
                run_id TEXT NOT NULL,
                score_id TEXT NOT NULL,
                objective_id TEXT NOT NULL,
                objective TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                sensor_frame_id TEXT NOT NULL,
                rollout_id TEXT,
                example_id TEXT NOT NULL,
                seed INTEGER NOT NULL,
                split TEXT NOT NULL,
                evaluation_stage TEXT NOT NULL,
                source TEXT NOT NULL,
                value REAL NOT NULL,
                rationale TEXT,
                metadata_json TEXT NOT NULL,
                score_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, score_id),
                FOREIGN KEY(run_id, objective_id) REFERENCES objectives(run_id, objective_id) ON DELETE CASCADE,
                FOREIGN KEY(run_id, sensor_frame_id) REFERENCES sensor_frames(run_id, sensor_frame_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS score_vectors (
                run_id TEXT NOT NULL,
                score_vector_id TEXT NOT NULL,
                objective_set_id TEXT NOT NULL,
                objective_set_hash TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                split TEXT NOT NULL,
                evaluation_stage TEXT NOT NULL,
                status TEXT NOT NULL,
                selection_objective TEXT NOT NULL,
                selection_score REAL,
                mean_reward REAL,
                score_count INTEGER NOT NULL,
                objective_values_json TEXT NOT NULL,
                covered_objectives_json TEXT NOT NULL,
                missing_objectives_json TEXT NOT NULL,
                example_ids_json TEXT NOT NULL,
                seeds_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                score_vector_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, score_vector_id),
                FOREIGN KEY(run_id, objective_set_id) REFERENCES objective_sets(run_id, objective_set_id) ON DELETE CASCADE,
                FOREIGN KEY(run_id, candidate_id) REFERENCES candidates(run_id, candidate_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS pareto_comparisons (
                run_id TEXT NOT NULL,
                pareto_comparison_id TEXT NOT NULL,
                objective_set_id TEXT NOT NULL,
                objective_set_hash TEXT NOT NULL,
                frontier_type TEXT NOT NULL,
                split TEXT NOT NULL,
                evaluation_stage TEXT NOT NULL,
                challenger_candidate_id TEXT NOT NULL,
                incumbent_candidate_id TEXT NOT NULL,
                challenger_score_vector_id TEXT NOT NULL,
                incumbent_score_vector_id TEXT NOT NULL,
                result TEXT NOT NULL,
                dominance_json TEXT NOT NULL,
                rationale TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                comparison_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, pareto_comparison_id),
                FOREIGN KEY(run_id, objective_set_id) REFERENCES objective_sets(run_id, objective_set_id) ON DELETE CASCADE,
                FOREIGN KEY(run_id, challenger_score_vector_id) REFERENCES score_vectors(run_id, score_vector_id) ON DELETE CASCADE,
                FOREIGN KEY(run_id, incumbent_score_vector_id) REFERENCES score_vectors(run_id, score_vector_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS trace_annotations (
                run_id TEXT NOT NULL,
                annotation_id TEXT NOT NULL,
                sensor_frame_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                rollout_id TEXT,
                example_id TEXT NOT NULL,
                evaluation_stage TEXT NOT NULL,
                backend TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL,
                trace_sha256 TEXT,
                event_count INTEGER NOT NULL,
                llm_request_count INTEGER NOT NULL,
                tool_call_count INTEGER NOT NULL,
                call_site_ids_json TEXT NOT NULL,
                support_count INTEGER NOT NULL,
                confidence REAL NOT NULL,
                annotation_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, annotation_id),
                FOREIGN KEY(run_id, sensor_frame_id) REFERENCES sensor_frames(run_id, sensor_frame_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS evidence_frames (
                run_id TEXT NOT NULL,
                evidence_frame_id TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                candidate_id TEXT,
                sensor_frame_id TEXT,
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                summary TEXT NOT NULL,
                score REAL,
                severity TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                frame_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, evidence_frame_id),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS verifier_jobs (
                run_id TEXT NOT NULL,
                verifier_job_id TEXT NOT NULL,
                verifier_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                sensor_frame_id TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                status TEXT NOT NULL,
                score REAL,
                severity TEXT NOT NULL,
                summary TEXT NOT NULL,
                result_json TEXT NOT NULL,
                failure_json TEXT,
                evidence_frame_ids_json TEXT NOT NULL,
                job_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, verifier_job_id),
                FOREIGN KEY(run_id, sensor_frame_id) REFERENCES sensor_frames(run_id, sensor_frame_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS subagent_invocations (
                run_id TEXT NOT NULL,
                invocation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                backend TEXT NOT NULL,
                trigger TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                sensor_frame_id TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                status TEXT NOT NULL,
                input_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                usage_json TEXT NOT NULL,
                cost_usd REAL NOT NULL,
                failure_json TEXT,
                invocation_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, invocation_id),
                FOREIGN KEY(run_id, sensor_frame_id) REFERENCES sensor_frames(run_id, sensor_frame_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS projection_freshness (
                run_id TEXT NOT NULL,
                projection_id TEXT NOT NULL,
                projection_name TEXT NOT NULL,
                status TEXT NOT NULL,
                source_table TEXT NOT NULL,
                source_count INTEGER NOT NULL,
                projected_table TEXT NOT NULL,
                projected_count INTEGER NOT NULL,
                lag_count INTEGER NOT NULL,
                checked_at TEXT NOT NULL,
                details_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, projection_id)
            );

            CREATE TABLE IF NOT EXISTS invariant_reports (
                run_id TEXT NOT NULL,
                report_id TEXT NOT NULL,
                status TEXT NOT NULL,
                violation_count INTEGER NOT NULL,
                checked_at TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                report_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, report_id)
            );

            CREATE TABLE IF NOT EXISTS invariant_violations (
                run_id TEXT NOT NULL,
                report_id TEXT NOT NULL,
                violation_id TEXT NOT NULL,
                invariant_id TEXT NOT NULL,
                severity TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                message TEXT NOT NULL,
                repair_hint TEXT,
                details_json TEXT NOT NULL,
                violation_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(run_id, report_id, violation_id),
                FOREIGN KEY(run_id, report_id) REFERENCES invariant_reports(run_id, report_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS artifact_refs (
                run_id TEXT NOT NULL,
                path TEXT NOT NULL,
                kind TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                bytes INTEGER NOT NULL,
                retention TEXT NOT NULL,
                PRIMARY KEY(run_id, path),
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS manifests (
                run_id TEXT PRIMARY KEY,
                manifest_path TEXT NOT NULL,
                best_candidate_id TEXT NOT NULL,
                cost_usd REAL NOT NULL,
                usage_json TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES optimization_runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_resolved_run_configs_run_algorithm
            ON resolved_run_configs(run_id, algorithm_id);

            CREATE INDEX IF NOT EXISTS idx_container_contract_snapshots_run_kind
            ON container_contract_snapshots(run_id, contract_kind, validation_status);

            CREATE INDEX IF NOT EXISTS idx_prompt_program_snapshots_run_program
            ON prompt_program_snapshots(run_id, program_id);

            CREATE INDEX IF NOT EXISTS idx_dataset_snapshots_run_split
            ON dataset_snapshots(run_id, split);

            CREATE INDEX IF NOT EXISTS idx_run_limits_run_policy
            ON run_limits(run_id, stop_policy);

            CREATE INDEX IF NOT EXISTS idx_rendered_optimizer_states_run_sequence
            ON rendered_optimizer_states(run_id, sequence_number);

            CREATE INDEX IF NOT EXISTS idx_rendered_optimizer_states_run_phase
            ON rendered_optimizer_states(run_id, run_phase, block_status);

            CREATE INDEX IF NOT EXISTS idx_runtime_effects_run_status
            ON runtime_effects(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_runtime_effects_run_lane
            ON runtime_effects(run_id, lane, status);

            CREATE INDEX IF NOT EXISTS idx_runtime_effects_run_subject
            ON runtime_effects(run_id, subject_type, subject_id);

            CREATE INDEX IF NOT EXISTS idx_runtime_effect_admissions_run_status
            ON runtime_effect_admissions(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_runtime_effect_admissions_run_effect
            ON runtime_effect_admissions(run_id, runtime_effect_id);

            CREATE INDEX IF NOT EXISTS idx_budget_reservations_run_status
            ON budget_reservations(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_budget_commits_run_reservation
            ON budget_commits(run_id, budget_reservation_id);

            CREATE INDEX IF NOT EXISTS idx_budget_releases_run_reservation
            ON budget_releases(run_id, budget_reservation_id);

            CREATE INDEX IF NOT EXISTS idx_candidates_run_status
            ON candidates(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_candidate_payloads_run_candidate
            ON candidate_payloads(run_id, candidate_id);

            CREATE INDEX IF NOT EXISTS idx_candidate_deltas_run_parent
            ON candidate_deltas(run_id, parent_candidate_id);

            CREATE INDEX IF NOT EXISTS idx_acceptance_decisions_run_decision
            ON acceptance_decisions(run_id, decision);

            CREATE INDEX IF NOT EXISTS idx_frontier_cells_run_status
            ON frontier_cells(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_plan_links_run_relation
            ON plan_links(run_id, relation);

            CREATE INDEX IF NOT EXISTS idx_cache_profiles_run_mode
            ON cache_profiles(run_id, mode);

            CREATE INDEX IF NOT EXISTS idx_cache_accesses_run_action
            ON cache_accesses(run_id, action);

            CREATE INDEX IF NOT EXISTS idx_cache_accesses_run_boundary
            ON cache_accesses(run_id, boundary);

            CREATE INDEX IF NOT EXISTS idx_materializations_run_candidate
            ON materializations(run_id, candidate_id, evaluation_stage, example_id);

            CREATE INDEX IF NOT EXISTS idx_materializations_run_cache_key
            ON materializations(run_id, cache_key);

            CREATE INDEX IF NOT EXISTS idx_materializations_run_status
            ON materializations(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_evaluation_cache_run_candidate_example
            ON evaluation_cache(run_id, candidate_hash, example_id, evaluator_id);

            CREATE INDEX IF NOT EXISTS idx_evaluation_cache_run_materializer
            ON evaluation_cache(run_id, algorithm_id, materializer_id, lever_version, sensor_version, objective_set_hash);

            CREATE INDEX IF NOT EXISTS idx_evaluation_cache_run_status
            ON evaluation_cache(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_event_stream_events_run_type
            ON event_stream_events(run_id, event_type);

            CREATE INDEX IF NOT EXISTS idx_usage_ledger_run_boundary
            ON usage_ledger(run_id, boundary);

            CREATE INDEX IF NOT EXISTS idx_usage_ledger_run_source
            ON usage_ledger(run_id, source_type, source_id);

            CREATE INDEX IF NOT EXISTS idx_stopper_states_run_status
            ON stopper_states(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_stopper_states_run_sequence
            ON stopper_states(run_id, sequence_number);

            CREATE INDEX IF NOT EXISTS idx_checkpoints_run_status
            ON checkpoints(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_checkpoints_run_kind
            ON checkpoints(run_id, checkpoint_kind);

            CREATE INDEX IF NOT EXISTS idx_checkpoints_run_sequence
            ON checkpoints(run_id, sequence_number);

            CREATE INDEX IF NOT EXISTS idx_optimizer_jobs_run_kind_status
            ON optimizer_jobs(run_id, kind, status);

            CREATE INDEX IF NOT EXISTS idx_optimizer_jobs_run_sensor
            ON optimizer_jobs(run_id, sensor_frame_id);

            CREATE INDEX IF NOT EXISTS idx_run_requests_status_priority
            ON run_requests(status, priority, submitted_at);

            CREATE INDEX IF NOT EXISTS idx_run_requests_run_id
            ON run_requests(run_id);

            CREATE INDEX IF NOT EXISTS idx_operations_run_status
            ON operations(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_operations_subject
            ON operations(subject_type, subject_id);

            CREATE INDEX IF NOT EXISTS idx_resource_leases_run_status
            ON resource_leases(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_resource_leases_kind_status
            ON resource_leases(resource_kind, status);

            CREATE INDEX IF NOT EXISTS idx_resource_leases_subject
            ON resource_leases(subject_type, subject_id);

            CREATE INDEX IF NOT EXISTS idx_rollout_jobs_run_status
            ON rollout_jobs(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_sensor_frames_run_candidate
            ON sensor_frames(run_id, candidate_id);

            CREATE INDEX IF NOT EXISTS idx_rollouts_run_status
            ON rollouts(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_rollouts_run_stage
            ON rollouts(run_id, evaluation_stage);

            CREATE INDEX IF NOT EXISTS idx_rollout_events_run_type
            ON rollout_events(run_id, event_type);

            CREATE INDEX IF NOT EXISTS idx_rollout_events_run_kind
            ON rollout_events(run_id, kind);

            CREATE INDEX IF NOT EXISTS idx_objective_sets_run_hash
            ON objective_sets(run_id, objective_set_hash);

            CREATE INDEX IF NOT EXISTS idx_scores_run_objective
            ON scores(run_id, objective);

            CREATE INDEX IF NOT EXISTS idx_scores_run_stage
            ON scores(run_id, evaluation_stage);

            CREATE INDEX IF NOT EXISTS idx_score_vectors_run_candidate
            ON score_vectors(run_id, candidate_id, split, evaluation_stage);

            CREATE INDEX IF NOT EXISTS idx_score_vectors_run_objective_set
            ON score_vectors(run_id, objective_set_hash);

            CREATE INDEX IF NOT EXISTS idx_pareto_comparisons_run_result
            ON pareto_comparisons(run_id, frontier_type, result);

            CREATE INDEX IF NOT EXISTS idx_pareto_comparisons_run_candidates
            ON pareto_comparisons(run_id, challenger_candidate_id, incumbent_candidate_id);

            CREATE INDEX IF NOT EXISTS idx_trace_annotations_run_status
            ON trace_annotations(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_evidence_frames_run_kind
            ON evidence_frames(run_id, kind);

            CREATE INDEX IF NOT EXISTS idx_verifier_jobs_run_status
            ON verifier_jobs(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_subagent_invocations_run_status
            ON subagent_invocations(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_projection_freshness_run_status
            ON projection_freshness(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_invariant_reports_run_status
            ON invariant_reports(run_id, status);

            CREATE INDEX IF NOT EXISTS idx_invariant_violations_run_severity
            ON invariant_violations(run_id, severity);
            "#,
        )?;
        self.ensure_run_request_schema()?;
        self.ensure_runtime_schema()?;
        self.ensure_runtime_indexes()?;
        Ok(())
    }

    fn ensure_run_request_schema(&self) -> Result<()> {
        self.ensure_column("run_requests", "run_workspace_db_path", "TEXT")?;
        self.ensure_column("run_requests", "result_manifest_path", "TEXT")?;
        self.ensure_column("run_requests", "best_candidate_id", "TEXT")?;
        self.ensure_column("run_requests", "cost_usd", "REAL")?;
        self.ensure_column("run_requests", "usage_json", "TEXT")?;
        self.ensure_column("run_requests", "result_json", "TEXT")?;
        Ok(())
    }

    fn ensure_runtime_schema(&self) -> Result<()> {
        self.ensure_column(
            "budget_commits",
            "wall_seconds",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        self.ensure_column("rollout_events", "kind", "TEXT NOT NULL DEFAULT ''")?;
        self.ensure_column(
            "rollout_events",
            "payload_json",
            "TEXT NOT NULL DEFAULT '{}'",
        )?;
        self.ensure_column("rollout_events", "trace_ref", "TEXT")?;
        self.ensure_column("optimizer_jobs", "worker_id", "TEXT")?;
        self.ensure_column("optimizer_jobs", "leased_at", "TEXT")?;
        self.ensure_column("optimizer_jobs", "lease_expires_at", "TEXT")?;
        self.ensure_column("optimizer_jobs", "heartbeat_at", "TEXT")?;
        self.ensure_column("optimizer_jobs", "next_retry_at", "TEXT")?;
        Ok(())
    }

    fn ensure_runtime_indexes(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            CREATE INDEX IF NOT EXISTS idx_optimizer_jobs_run_lease
            ON optimizer_jobs(run_id, status, lease_expires_at);

            CREATE INDEX IF NOT EXISTS idx_optimizer_jobs_run_retry
            ON optimizer_jobs(run_id, status, next_retry_at);
            "#,
        )?;
        Ok(())
    }

    fn ensure_column(&self, table: &str, column: &str, column_type: &str) -> Result<()> {
        let mut stmt = self.conn.prepare(&format!("PRAGMA table_info({table})"))?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let existing: String = row.get(1)?;
            if existing == column {
                return Ok(());
            }
        }
        drop(rows);
        drop(stmt);
        self.conn.execute(
            &format!("ALTER TABLE {table} ADD COLUMN {column} {column_type}"),
            [],
        )?;
        Ok(())
    }

    fn run_requests(&self) -> Result<Vec<WorkspaceRunRequestStatus>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT request_id, run_id, status, config_path, container_url,
                   cache_mode, cache_namespace, output_dir, run_dir, priority,
                   submitted_at, leased_at, lease_expires_at, started_at,
                   finished_at, updated_at, lease_id, worker_id,
                   run_workspace_db_path, result_manifest_path,
                   best_candidate_id, cost_usd, usage_json, result_json,
                   error_json
            FROM run_requests
            ORDER BY submitted_at, request_id
            "#,
        )?;
        let mut rows = stmt.query([])?;
        let mut requests = Vec::new();
        while let Some(row) = rows.next()? {
            requests.push(run_request_from_row(row)?);
        }
        Ok(requests)
    }

    fn load_run_request(&self, request_id: &str) -> Result<WorkspaceRunRequestStatus> {
        self.conn
            .query_row(
                r#"
                SELECT request_id, run_id, status, config_path, container_url,
                       cache_mode, cache_namespace, output_dir, run_dir,
                       priority, submitted_at, leased_at, lease_expires_at,
                       started_at, finished_at, updated_at, lease_id,
                       worker_id, run_workspace_db_path, result_manifest_path,
                       best_candidate_id, cost_usd, usage_json, result_json,
                       error_json
                FROM run_requests
                WHERE request_id = ?1
                "#,
                params![request_id],
                run_request_from_row,
            )
            .optional()?
            .ok_or_else(|| {
                OptimizerError::Config(format!("run request does not exist: {request_id}"))
            })
    }

    fn update_run_request_status<P>(
        &self,
        request_id: &str,
        target_status: &str,
        sql: &str,
        params: P,
    ) -> Result<WorkspaceRunRequestStatus>
    where
        P: rusqlite::Params,
    {
        let updated = self.conn.execute(sql, params)?;
        if updated == 0 {
            return Err(OptimizerError::Config(format!(
                "run request does not exist: {request_id}"
            )));
        }
        let request = self.run_request(request_id)?;
        if request.status != target_status {
            return Err(OptimizerError::Config(format!(
                "run request {request_id} did not transition to {target_status}"
            )));
        }
        Ok(request)
    }

    fn record_run_request_operation(
        &self,
        operation_type: &str,
        request: &WorkspaceRunRequestStatus,
        status: &str,
        idempotency_key: Option<String>,
        metadata: Map<String, Value>,
    ) -> Result<()> {
        let operation = OperationRecord::run_request(
            operation_type,
            &request.run_id,
            &request.request_id,
            status,
            idempotency_key,
            metadata,
        );
        self.conn.execute(
            r#"
            INSERT INTO operations(
                operation_id, operation_type, subject_type, subject_id, run_id,
                idempotency_key, status, attempt, metadata_json, operation_json,
                updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, datetime('now'))
            ON CONFLICT(operation_id) DO UPDATE SET
                operation_type = excluded.operation_type,
                subject_type = excluded.subject_type,
                subject_id = excluded.subject_id,
                run_id = excluded.run_id,
                idempotency_key = excluded.idempotency_key,
                status = excluded.status,
                attempt = excluded.attempt,
                metadata_json = excluded.metadata_json,
                operation_json = excluded.operation_json,
                updated_at = datetime('now')
            "#,
            params![
                operation.operation_id,
                operation.operation_type,
                operation.subject_type,
                operation.subject_id,
                operation.run_id,
                operation.idempotency_key,
                operation.status,
                operation.attempt as i64,
                stable_json(&serde_json::to_value(&operation.metadata)?),
                stable_json(&serde_json::to_value(&operation)?),
            ],
        )?;
        Ok(())
    }

    fn record_resource_leases_for_request(
        &self,
        request: &WorkspaceRunRequestStatus,
    ) -> Result<()> {
        let Some(lease_id) = request.lease_id.as_deref() else {
            return Ok(());
        };
        let mut resources = Vec::new();
        if !request.container_url.trim().is_empty() {
            resources.push(("container_url", request.container_url.as_str()));
        }
        if !request.cache_namespace.trim().is_empty() {
            resources.push(("cache_namespace", request.cache_namespace.as_str()));
        }
        for (kind, resource_id) in resources {
            let mut metadata = Map::new();
            if let Some(worker_id) = &request.worker_id {
                metadata.insert("worker_id".to_string(), Value::String(worker_id.clone()));
            }
            let lease = ResourceLeaseRecord::run_request(ResourceLeaseRecordInput {
                lease_id,
                resource_kind: kind,
                resource_id,
                run_id: &request.run_id,
                request_id: &request.request_id,
                status: "active",
                lease_expires_at: request.lease_expires_at.clone(),
                metadata,
            });
            self.conn.execute(
                r#"
                INSERT INTO resource_leases(
                    resource_lease_id, lease_id, resource_kind, resource_id,
                    subject_type, subject_id, run_id, status, lease_expires_at,
                    metadata_json, lease_json, updated_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, datetime('now'))
                ON CONFLICT(resource_lease_id) DO UPDATE SET
                    lease_id = excluded.lease_id,
                    resource_kind = excluded.resource_kind,
                    resource_id = excluded.resource_id,
                    subject_type = excluded.subject_type,
                    subject_id = excluded.subject_id,
                    run_id = excluded.run_id,
                    status = excluded.status,
                    lease_expires_at = excluded.lease_expires_at,
                    metadata_json = excluded.metadata_json,
                    lease_json = excluded.lease_json,
                    updated_at = datetime('now')
                "#,
                params![
                    lease.resource_lease_id,
                    lease.lease_id,
                    lease.resource_kind,
                    lease.resource_id,
                    lease.subject_type,
                    lease.subject_id,
                    lease.run_id,
                    lease.status,
                    lease.lease_expires_at,
                    stable_json(&serde_json::to_value(&lease.metadata)?),
                    stable_json(&serde_json::to_value(&lease)?),
                ],
            )?;
        }
        Ok(())
    }

    fn update_resource_leases_for_request(&self, request_id: &str, status: &str) -> Result<()> {
        self.conn.execute(
            r#"
            UPDATE resource_leases
            SET status = ?2,
                updated_at = datetime('now')
            WHERE subject_type = 'run_request'
              AND subject_id = ?1
              AND status = 'active'
            "#,
            params![request_id, status],
        )?;
        Ok(())
    }

    fn refresh_workspace_health(&self) -> Result<()> {
        let checked_at = now_rfc3339();
        self.refresh_global_workspace_health(&checked_at)?;
        let mut stmt = self.conn.prepare(
            "SELECT run_id, state, best_candidate_id FROM optimization_runs ORDER BY run_id",
        )?;
        let mut rows = stmt.query([])?;
        let mut runs = Vec::new();
        while let Some(row) = rows.next()? {
            runs.push((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
            ));
        }
        drop(rows);
        drop(stmt);

        for (run_id, state, best_candidate_id) in runs {
            let counts = self.run_health_counts(&run_id)?;
            let projections = run_projection_freshness(&run_id, &state, &counts, &checked_at);
            self.conn.execute(
                "DELETE FROM projection_freshness WHERE run_id = ?1",
                params![run_id],
            )?;
            for projection in projections {
                self.upsert_projection_freshness(&projection)?;
            }
            let report = build_run_invariant_report(
                &run_id,
                &state,
                best_candidate_id,
                &counts,
                &checked_at,
            );
            self.upsert_invariant_report(&report)?;
        }
        Ok(())
    }

    fn refresh_global_workspace_health(&self, checked_at: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM projection_freshness WHERE run_id = '__workspace__'",
            [],
        )?;
        let run_requests = self.scalar_count("SELECT COUNT(*) FROM run_requests")?;
        let submit_operations = self.scalar_count(
            "SELECT COUNT(*) FROM operations WHERE operation_type = 'submit_run_request'",
        )?;
        let projection = ProjectionFreshnessRecord::exact_count(
            "__workspace__",
            "run_request_submit_operations",
            "run_requests",
            run_requests,
            "operations.submit_run_request",
            submit_operations,
            checked_at,
        );
        self.upsert_projection_freshness(&projection)?;

        let mut violations = Vec::new();
        let active_without_lease = self.scalar_count(
            "SELECT COUNT(*) FROM run_requests WHERE status IN ('leased', 'running') AND lease_id IS NULL",
        )?;
        if active_without_lease > 0 {
            violations.push(InvariantViolation::new(InvariantViolationInput {
                run_id: "__workspace__",
                invariant_id: "active_requests_have_lease",
                severity: "error",
                subject_type: "run_requests",
                subject_id: "active",
                message: "active run requests must have a lease id".to_string(),
                repair_hint: Some(
                    "recover expired requests or reclaim them with a lease id".to_string(),
                ),
                details: json!({"active_without_lease": active_without_lease}),
            }));
        }
        let terminal_active_leases = self.scalar_count(
            r#"
            SELECT COUNT(*)
            FROM resource_leases AS leases
            JOIN run_requests AS requests
              ON requests.request_id = leases.subject_id
            WHERE leases.status = 'active'
              AND requests.status IN ('completed', 'failed', 'cancelled')
            "#,
        )?;
        if terminal_active_leases > 0 {
            violations.push(InvariantViolation::new(InvariantViolationInput {
                run_id: "__workspace__",
                invariant_id: "terminal_requests_release_leases",
                severity: "error",
                subject_type: "resource_leases",
                subject_id: "active",
                message: "terminal run requests must not keep active resource leases".to_string(),
                repair_hint: Some(
                    "release or expire active leases for terminal requests".to_string(),
                ),
                details: json!({"terminal_active_leases": terminal_active_leases}),
            }));
        }
        let orphan_active_leases = self.scalar_count(
            r#"
            SELECT COUNT(*)
            FROM resource_leases AS leases
            LEFT JOIN run_requests AS requests
              ON requests.request_id = leases.subject_id
            WHERE leases.status = 'active'
              AND leases.subject_type = 'run_request'
              AND (
                requests.request_id IS NULL
                OR requests.status NOT IN ('leased', 'running')
              )
            "#,
        )?;
        if orphan_active_leases > 0 {
            violations.push(InvariantViolation::new(InvariantViolationInput {
                run_id: "__workspace__",
                invariant_id: "active_leases_point_to_active_requests",
                severity: "error",
                subject_type: "resource_leases",
                subject_id: "active",
                message: "active resource leases must point to leased or running requests"
                    .to_string(),
                repair_hint: Some("recover expired requests and expire orphan leases".to_string()),
                details: json!({"orphan_active_leases": orphan_active_leases}),
            }));
        }
        let report = InvariantReport::new(
            "__workspace__",
            checked_at,
            violations,
            json!({
                "run_requests": run_requests,
                "submit_operations": submit_operations,
            }),
        );
        self.upsert_invariant_report(&report)?;
        Ok(())
    }

    fn run_health_counts(&self, run_id: &str) -> Result<RunHealthCounts> {
        Ok(RunHealthCounts {
            resolved_run_configs: self.count_where("resolved_run_configs", run_id)?,
            container_contract_snapshots: self.count_where(
                "container_contract_snapshots",
                run_id,
            )?,
            prompt_program_snapshots: self.count_where("prompt_program_snapshots", run_id)?,
            dataset_snapshots: self.count_where("dataset_snapshots", run_id)?,
            run_limits: self.count_where("run_limits", run_id)?,
            rendered_optimizer_states: self.count_where("rendered_optimizer_states", run_id)?,
            runtime_effects: self.count_where("runtime_effects", run_id)?,
            runtime_effect_admissions: self.count_where("runtime_effect_admissions", run_id)?,
            budget_reservations: self.count_where("budget_reservations", run_id)?,
            budget_commits: self.count_where("budget_commits", run_id)?,
            budget_releases: self.count_where("budget_releases", run_id)?,
            active_budget_reservations: self.count_run_query(
                r#"
                SELECT COUNT(*)
                FROM budget_reservations
                WHERE run_id = ?1
                  AND status IN ('reserved', 'planned', 'active', 'leased', 'running')
                "#,
                run_id,
            )?,
            candidates: self.count_where("candidates", run_id)?,
            parented_candidates: self.count_run_query(
                "SELECT COUNT(*) FROM candidates WHERE run_id = ?1 AND parent_id IS NOT NULL",
                run_id,
            )?,
            train_frontier_candidates: self.count_run_query(
                r#"
                SELECT COUNT(*)
                FROM candidates
                WHERE run_id = ?1
                  AND train_reward IS NOT NULL
                  AND train_reward = (
                    SELECT MAX(train_reward)
                    FROM candidates
                    WHERE run_id = ?1 AND train_reward IS NOT NULL
                  )
                "#,
                run_id,
            )?,
            candidate_payloads: self.count_where("candidate_payloads", run_id)?,
            candidate_deltas: self.count_where("candidate_deltas", run_id)?,
            acceptance_decisions: self.count_where("acceptance_decisions", run_id)?,
            frontier_cells: self.count_where("frontier_cells", run_id)?,
            plan_links: self.count_where("plan_links", run_id)?,
            cache_profiles: self.count_where("cache_profiles", run_id)?,
            cache_accesses: self.count_where("cache_accesses", run_id)?,
            materializations: self.count_where("materializations", run_id)?,
            evaluation_cache: self.count_where("evaluation_cache", run_id)?,
            evaluation_cache_expected: self.count_run_query(
                r#"
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT COALESCE(platform_cache_key, cache_key) AS cache_key
                    FROM materializations
                    WHERE run_id = ?1
                      AND COALESCE(platform_cache_key, cache_key) IS NOT NULL
                )
                "#,
                run_id,
            )?,
            cache_profile_access_total: self.count_run_query(
                "SELECT COALESCE(MAX(total_accesses), 0) FROM cache_profiles WHERE run_id = ?1",
                run_id,
            )?,
            event_stream_events: self.count_where("event_stream_events", run_id)?,
            usage_ledger: self.count_where("usage_ledger", run_id)?,
            stopper_states: self.count_where("stopper_states", run_id)?,
            checkpoints: self.count_where("checkpoints", run_id)?,
            usage_ledger_expected: self.count_run_query(
                r#"
                SELECT
                  (SELECT COUNT(*) FROM sensor_frames WHERE run_id = ?1)
                  +
                  (SELECT COUNT(*) FROM event_stream_events
                   WHERE run_id = ?1 AND event_type = 'proposer.completed')
                "#,
                run_id,
            )?,
            rollout_jobs: self.count_where("rollout_jobs", run_id)?,
            sensor_frames: self.count_where("sensor_frames", run_id)?,
            rollouts: self.count_where("rollouts", run_id)?,
            rollout_events: self.count_where("rollout_events", run_id)?,
            objective_sets: self.count_where("objective_sets", run_id)?,
            objectives: self.count_where("objectives", run_id)?,
            scores: self.count_where("scores", run_id)?,
            score_vector_sources: self.count_run_query(
                "SELECT COUNT(*) FROM (SELECT DISTINCT candidate_id, split, evaluation_stage FROM scores WHERE run_id = ?1)",
                run_id,
            )?,
            score_vectors: self.count_where("score_vectors", run_id)?,
            pareto_comparisons: self.count_where("pareto_comparisons", run_id)?,
            trace_annotations: self.count_where("trace_annotations", run_id)?,
            evidence_frames: self.count_where("evidence_frames", run_id)?,
            verifier_jobs: self.count_where("verifier_jobs", run_id)?,
            subagent_invocations: self.count_where("subagent_invocations", run_id)?,
            manifests: self.count_where("manifests", run_id)?,
            state_transitions: self.count_where("optimizer_state_history", run_id)?,
            active_resource_leases: self.count_run_query(
                "SELECT COUNT(*) FROM resource_leases WHERE run_id = ?1 AND status = 'active'",
                run_id,
            )?,
        })
    }

    fn upsert_projection_freshness(&self, record: &ProjectionFreshnessRecord) -> Result<()> {
        self.conn.execute(
            r#"
            INSERT INTO projection_freshness(
                run_id, projection_id, projection_name, status, source_table,
                source_count, projected_table, projected_count, lag_count,
                checked_at, details_json, record_json, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, datetime('now'))
            ON CONFLICT(run_id, projection_id) DO UPDATE SET
                projection_name = excluded.projection_name,
                status = excluded.status,
                source_table = excluded.source_table,
                source_count = excluded.source_count,
                projected_table = excluded.projected_table,
                projected_count = excluded.projected_count,
                lag_count = excluded.lag_count,
                checked_at = excluded.checked_at,
                details_json = excluded.details_json,
                record_json = excluded.record_json,
                updated_at = datetime('now')
            "#,
            params![
                record.run_id,
                record.projection_id,
                record.projection_name,
                record.status,
                record.source_table,
                record.source_count as i64,
                record.projected_table,
                record.projected_count as i64,
                record.lag_count as i64,
                record.checked_at,
                stable_json(&record.details),
                stable_json(&serde_json::to_value(record)?),
            ],
        )?;
        Ok(())
    }

    fn upsert_invariant_report(&self, report: &InvariantReport) -> Result<()> {
        self.conn.execute(
            "DELETE FROM invariant_violations WHERE run_id = ?1 AND report_id = ?2",
            params![report.run_id, report.report_id],
        )?;
        self.conn.execute(
            r#"
            INSERT INTO invariant_reports(
                run_id, report_id, status, violation_count, checked_at,
                summary_json, report_json, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, datetime('now'))
            ON CONFLICT(run_id, report_id) DO UPDATE SET
                status = excluded.status,
                violation_count = excluded.violation_count,
                checked_at = excluded.checked_at,
                summary_json = excluded.summary_json,
                report_json = excluded.report_json,
                updated_at = datetime('now')
            "#,
            params![
                report.run_id,
                report.report_id,
                report.status,
                report.violation_count as i64,
                report.checked_at,
                stable_json(&report.summary),
                stable_json(&serde_json::to_value(report)?),
            ],
        )?;
        for violation in &report.violations {
            self.conn.execute(
                r#"
                INSERT INTO invariant_violations(
                    run_id, report_id, violation_id, invariant_id, severity,
                    subject_type, subject_id, message, repair_hint, details_json,
                    violation_json, updated_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, datetime('now'))
                ON CONFLICT(run_id, report_id, violation_id) DO UPDATE SET
                    invariant_id = excluded.invariant_id,
                    severity = excluded.severity,
                    subject_type = excluded.subject_type,
                    subject_id = excluded.subject_id,
                    message = excluded.message,
                    repair_hint = excluded.repair_hint,
                    details_json = excluded.details_json,
                    violation_json = excluded.violation_json,
                    updated_at = datetime('now')
                "#,
                params![
                    violation.run_id,
                    report.report_id,
                    violation.violation_id,
                    violation.invariant_id,
                    violation.severity,
                    violation.subject_type,
                    violation.subject_id,
                    violation.message,
                    violation.repair_hint,
                    stable_json(&violation.details),
                    stable_json(&serde_json::to_value(violation)?),
                ],
            )?;
        }
        Ok(())
    }

    fn projection_freshness_for_run(&self, run_id: &str) -> Result<Vec<ProjectionFreshnessRecord>> {
        let mut stmt = self.conn.prepare(
            r#"
            SELECT record_json
            FROM projection_freshness
            WHERE run_id = ?1
            ORDER BY projection_name
            "#,
        )?;
        let mut rows = stmt.query(params![run_id])?;
        let mut records = Vec::new();
        while let Some(row) = rows.next()? {
            let raw: String = row.get(0)?;
            records.push(serde_json::from_str(&raw)?);
        }
        Ok(records)
    }

    fn invariant_report_for_run(&self, run_id: &str) -> Result<Option<InvariantReport>> {
        self.conn
            .query_row(
                r#"
                SELECT report_json
                FROM invariant_reports
                WHERE run_id = ?1
                ORDER BY checked_at DESC
                LIMIT 1
                "#,
                params![run_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .map(|raw| serde_json::from_str(&raw).map_err(OptimizerError::from))
            .transpose()
    }

    fn entity_counts(&self, run_id: &str) -> Result<WorkspaceEntityCounts> {
        Ok(WorkspaceEntityCounts {
            resolved_run_configs: self.count_where("resolved_run_configs", run_id)?,
            container_contract_snapshots: self
                .count_where("container_contract_snapshots", run_id)?,
            prompt_program_snapshots: self.count_where("prompt_program_snapshots", run_id)?,
            dataset_snapshots: self.count_where("dataset_snapshots", run_id)?,
            run_limits: self.count_where("run_limits", run_id)?,
            rendered_optimizer_states: self.count_where("rendered_optimizer_states", run_id)?,
            runtime_effects: self.count_where("runtime_effects", run_id)?,
            runtime_effect_admissions: self.count_where("runtime_effect_admissions", run_id)?,
            budget_reservations: self.count_where("budget_reservations", run_id)?,
            budget_commits: self.count_where("budget_commits", run_id)?,
            budget_releases: self.count_where("budget_releases", run_id)?,
            candidates: self.count_where("candidates", run_id)?,
            candidate_payloads: self.count_where("candidate_payloads", run_id)?,
            candidate_deltas: self.count_where("candidate_deltas", run_id)?,
            acceptance_decisions: self.count_where("acceptance_decisions", run_id)?,
            frontier_cells: self.count_where("frontier_cells", run_id)?,
            plan_links: self.count_where("plan_links", run_id)?,
            cache_profiles: self.count_where("cache_profiles", run_id)?,
            cache_accesses: self.count_where("cache_accesses", run_id)?,
            materializations: self.count_where("materializations", run_id)?,
            evaluation_cache: self.count_where("evaluation_cache", run_id)?,
            event_stream_events: self.count_where("event_stream_events", run_id)?,
            usage_ledger: self.count_where("usage_ledger", run_id)?,
            stopper_states: self.count_where("stopper_states", run_id)?,
            checkpoints: self.count_where("checkpoints", run_id)?,
            optimizer_jobs: self.count_where("optimizer_jobs", run_id)?,
            rollout_jobs: self.count_where("rollout_jobs", run_id)?,
            rollouts: self.count_where("rollouts", run_id)?,
            rollout_events: self.count_where("rollout_events", run_id)?,
            operations: self.count_where("operations", run_id)?,
            resource_leases: self.count_where("resource_leases", run_id)?,
            objective_sets: self.count_where("objective_sets", run_id)?,
            objectives: self.count_where("objectives", run_id)?,
            scores: self.count_where("scores", run_id)?,
            score_vectors: self.count_where("score_vectors", run_id)?,
            pareto_comparisons: self.count_where("pareto_comparisons", run_id)?,
            sensor_frames: self.count_where("sensor_frames", run_id)?,
            trace_annotations: self.count_where("trace_annotations", run_id)?,
            evidence_frames: self.count_where("evidence_frames", run_id)?,
            verifier_jobs: self.count_where("verifier_jobs", run_id)?,
            subagent_invocations: self.count_where("subagent_invocations", run_id)?,
            projection_freshness: self.count_where("projection_freshness", run_id)?,
            invariant_reports: self.count_where("invariant_reports", run_id)?,
            invariant_violations: self.count_where("invariant_violations", run_id)?,
            state_transitions: self.count_where("optimizer_state_history", run_id)?,
            artifact_refs: self.count_where("artifact_refs", run_id)?,
            manifests: self.count_where("manifests", run_id)?,
        })
    }

    fn count_where(&self, table: &str, run_id: &str) -> Result<u64> {
        let sql = format!("SELECT COUNT(*) FROM {table} WHERE run_id = ?1");
        Ok(self
            .conn
            .query_row(&sql, params![run_id], |row| row.get::<_, i64>(0))?
            .max(0) as u64)
    }

    fn count_run_query(&self, sql: &str, run_id: &str) -> Result<u64> {
        Ok(self
            .conn
            .query_row(sql, params![run_id], |row| row.get::<_, i64>(0))?
            .max(0) as u64)
    }

    fn scalar_count(&self, sql: &str) -> Result<u64> {
        Ok(self
            .conn
            .query_row(sql, [], |row| row.get::<_, i64>(0))?
            .max(0) as u64)
    }

    fn group_counts(&self, sql: &str, run_id: &str) -> Result<BTreeMap<String, u64>> {
        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = stmt.query(params![run_id])?;
        let mut counts = BTreeMap::new();
        while let Some(row) = rows.next()? {
            let key: String = row.get(0)?;
            let count = row.get::<_, i64>(1)?.max(0) as u64;
            counts.insert(key, count);
        }
        Ok(counts)
    }

    fn global_group_counts(&self, sql: &str) -> Result<BTreeMap<String, u64>> {
        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = stmt.query([])?;
        let mut counts = BTreeMap::new();
        while let Some(row) = rows.next()? {
            let key: String = row.get(0)?;
            let count = row.get::<_, i64>(1)?.max(0) as u64;
            counts.insert(key, count);
        }
        Ok(counts)
    }

    pub fn latest_transition(
        &self,
        run_id: &str,
    ) -> Result<Option<WorkspaceStateTransitionStatus>> {
        self.conn
            .query_row(
                r#"
                SELECT sequence_number, from_state, to_state, trigger,
                       message, transition_at, details_json
                FROM optimizer_state_history
                WHERE run_id = ?1
                ORDER BY sequence_number DESC
                LIMIT 1
                "#,
                params![run_id],
                |row| {
                    let details_json = row.get::<_, String>(6)?;
                    Ok(WorkspaceStateTransitionStatus {
                        sequence_number: row.get::<_, i64>(0)?.max(0) as u64,
                        from_state: row.get(1)?,
                        to_state: row.get(2)?,
                        trigger: row.get(3)?,
                        message: row.get(4)?,
                        transition_at: row.get(5)?,
                        details: parse_json_or_null(Some(&details_json)),
                    })
                },
            )
            .optional()
            .map_err(OptimizerError::from)
    }

    fn latest_rendered_state(&self, run_id: &str) -> Result<Option<RenderedOptimizerStateRecord>> {
        self.conn
            .query_row(
                r#"
                SELECT record_json
                FROM rendered_optimizer_states
                WHERE run_id = ?1
                ORDER BY sequence_number DESC
                LIMIT 1
                "#,
                params![run_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .map(|raw| serde_json::from_str(&raw).map_err(OptimizerError::from))
            .transpose()
    }

    fn rendered_optimizer_state_for_transition(
        &self,
        sequence_number: usize,
        transition: &OptimizerTransition,
    ) -> Result<RenderedOptimizerStateRecord> {
        let run_id = transition.run_id.as_str();
        let active_effect_count = self.count_run_query(
            r#"
            SELECT COUNT(*)
            FROM runtime_effects
            WHERE run_id = ?1
              AND status NOT IN ('completed', 'failed', 'cancelled', 'canceled', 'expired', 'rejected')
            "#,
            run_id,
        )?;
        let active_optimizer_jobs = self.count_run_query(
            r#"
            SELECT COUNT(*)
            FROM optimizer_jobs
            WHERE run_id = ?1
              AND status NOT IN ('completed', 'failed', 'cancelled', 'expired')
            "#,
            run_id,
        )?;
        let active_rollout_jobs = self.count_run_query(
            r#"
            SELECT COUNT(*)
            FROM rollout_jobs
            WHERE run_id = ?1
              AND status NOT IN ('completed', 'failed', 'cancelled', 'expired')
            "#,
            run_id,
        )?;
        let active_job_count = active_optimizer_jobs + active_rollout_jobs;
        let best_candidate_id = transition
            .details
            .get("best_candidate_id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or(self.best_candidate_id(run_id)?);
        let frontier_size = self.frontier_size(run_id)?;
        let queue_counts = json!({
            "optimizer_jobs": {
                "by_status": self.group_counts(
                    "SELECT status, COUNT(*) FROM optimizer_jobs WHERE run_id = ?1 GROUP BY status",
                    run_id,
                )?,
                "by_kind": self.group_counts(
                    "SELECT kind, COUNT(*) FROM optimizer_jobs WHERE run_id = ?1 GROUP BY kind",
                    run_id,
                )?,
                "by_lease": self.group_counts(
                    r#"
                    SELECT
                        CASE
                            WHEN status IN ('completed', 'failed', 'cancelled', 'expired') THEN 'terminal'
                            WHEN lease_id IS NULL THEN 'unleased'
                            WHEN lease_expires_at IS NOT NULL AND lease_expires_at <= datetime('now') THEN 'lease_expired'
                            ELSE 'lease_active'
                        END AS lease_state,
                        COUNT(*)
                    FROM optimizer_jobs
                    WHERE run_id = ?1
                    GROUP BY lease_state
                    "#,
                    run_id,
                )?,
            },
            "rollout_jobs": {
                "by_status": self.group_counts(
                    "SELECT status, COUNT(*) FROM rollout_jobs WHERE run_id = ?1 GROUP BY status",
                    run_id,
                )?,
                "by_stage": self.group_counts(
                    "SELECT evaluation_stage, COUNT(*) FROM rollout_jobs WHERE run_id = ?1 GROUP BY evaluation_stage",
                    run_id,
                )?,
            },
            "runtime_effects": {
                "by_status": self.group_counts(
                    "SELECT status, COUNT(*) FROM runtime_effects WHERE run_id = ?1 GROUP BY status",
                    run_id,
                )?,
                "by_lane": self.group_counts(
                    "SELECT lane, COUNT(*) FROM runtime_effects WHERE run_id = ?1 GROUP BY lane",
                    run_id,
                )?,
                "failures_by_class": self.group_counts(
                    "SELECT failure_class, COUNT(*) FROM runtime_effects WHERE run_id = ?1 AND failure_class IS NOT NULL GROUP BY failure_class",
                    run_id,
                )?,
                "admissions_by_status": self.group_counts(
                    "SELECT status, COUNT(*) FROM runtime_effect_admissions WHERE run_id = ?1 GROUP BY status",
                    run_id,
                )?,
            },
            "budget_reservations": {
                "by_status": self.group_counts(
                    "SELECT status, COUNT(*) FROM budget_reservations WHERE run_id = ?1 GROUP BY status",
                    run_id,
                )?,
            },
            "active_effect_count": active_effect_count,
            "active_job_count": active_job_count,
        });
        let budget_status = self.rendered_budget_status(run_id)?;
        let evidence_status = json!({
            "artifact_refs": self.count_where("artifact_refs", run_id)?,
            "trace_annotations": self.count_where("trace_annotations", run_id)?,
            "evidence_frames": self.count_where("evidence_frames", run_id)?,
            "verifier_jobs": self.count_where("verifier_jobs", run_id)?,
            "subagent_invocations": self.count_where("subagent_invocations", run_id)?,
            "invariant_violations": self.count_where("invariant_violations", run_id)?,
            "manifests": self.count_where("manifests", run_id)?,
        });
        let terminal_status = rendered_terminal_status(transition.to.as_str());
        let block_status = rendered_block_status(
            transition.to.as_str(),
            active_effect_count,
            active_job_count,
        );
        let mut details = Map::new();
        details.insert("transition".to_string(), serde_json::to_value(transition)?);
        details.insert("trigger".to_string(), json!(transition.trigger.as_str()));
        details.insert("message".to_string(), json!(&transition.message));
        Ok(RenderedOptimizerStateRecord::from_input(
            RenderedOptimizerStateInput {
                run_id,
                sequence_number: sequence_number as u64,
                run_phase: transition.to.as_str(),
                generation_phase: transition
                    .details
                    .get("generation")
                    .map(|value| format!("generation_{value}")),
                candidate_phase: transition
                    .details
                    .get("stage")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                block_status,
                terminal_status,
                best_candidate_id,
                frontier_size,
                active_effect_count,
                active_job_count,
                queue_counts,
                budget_status,
                evidence_status,
                details: Value::Object(details),
            },
        ))
    }

    fn best_candidate_id(&self, run_id: &str) -> Result<Option<String>> {
        let explicit = self
            .conn
            .query_row(
                "SELECT best_candidate_id FROM optimization_runs WHERE run_id = ?1",
                params![run_id],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()?
            .flatten()
            .filter(|value| !value.trim().is_empty());
        if explicit.is_some() {
            return Ok(explicit);
        }
        self.conn
            .query_row(
                r#"
                SELECT candidate_id
                FROM candidates
                WHERE run_id = ?1 AND train_reward IS NOT NULL
                ORDER BY train_reward DESC, updated_at DESC, candidate_id DESC
                LIMIT 1
                "#,
                params![run_id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(OptimizerError::from)
    }

    fn frontier_size(&self, run_id: &str) -> Result<u64> {
        let explicit = self.count_run_query(
            "SELECT COUNT(*) FROM frontier_cells WHERE run_id = ?1 AND status != 'removed'",
            run_id,
        )?;
        if explicit > 0 {
            return Ok(explicit);
        }
        self.count_run_query(
            "SELECT COUNT(*) FROM candidates WHERE run_id = ?1 AND train_reward IS NOT NULL",
            run_id,
        )
    }

    fn rendered_budget_status(&self, run_id: &str) -> Result<Value> {
        Ok(serde_json::to_value(self.budget_ledger_snapshot(run_id)?)?)
    }
}

fn rendered_block_status(
    run_phase: &str,
    active_effect_count: u64,
    active_job_count: u64,
) -> &'static str {
    match run_phase {
        "paused" => "operator_pause",
        "completed" | "failed" | "cancelled" => "none",
        _ if active_job_count > 0 => "waiting_for_jobs",
        _ if active_effect_count > 0 => "waiting_for_effects",
        _ => "none",
    }
}

fn rendered_terminal_status(run_phase: &str) -> Option<String> {
    match run_phase {
        "completed" => Some("success".to_string()),
        "failed" => Some("failed".to_string()),
        "cancelled" => Some("cancelled".to_string()),
        _ => None,
    }
}

impl<'a> WorkspaceView<'a> {
    fn json_records<T>(&self, run_id: &str, sql: &str) -> Result<Vec<T>>
    where
        T: DeserializeOwned,
    {
        let mut stmt = self.store.conn.prepare(sql)?;
        let mut rows = stmt.query(params![run_id])?;
        let mut records = Vec::new();
        while let Some(row) = rows.next()? {
            let raw: String = row.get(0)?;
            records.push(serde_json::from_str(&raw)?);
        }
        Ok(records)
    }

    pub fn resolved_run_config_records(
        &self,
        run_id: &str,
    ) -> Result<Vec<ResolvedRunConfigRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT record_json
            FROM resolved_run_configs
            WHERE run_id = ?1
            ORDER BY recorded_at, resolved_config_id
            "#,
        )
    }

    pub fn container_contract_snapshot_records(
        &self,
        run_id: &str,
    ) -> Result<Vec<ContainerContractSnapshotRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT record_json
            FROM container_contract_snapshots
            WHERE run_id = ?1
            ORDER BY recorded_at, contract_kind, contract_snapshot_id
            "#,
        )
    }

    pub fn prompt_program_snapshot_records(
        &self,
        run_id: &str,
    ) -> Result<Vec<PromptProgramSnapshotRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT record_json
            FROM prompt_program_snapshots
            WHERE run_id = ?1
            ORDER BY recorded_at, program_id, program_snapshot_id
            "#,
        )
    }

    pub fn dataset_snapshot_records(&self, run_id: &str) -> Result<Vec<DatasetSnapshotRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT record_json
            FROM dataset_snapshots
            WHERE run_id = ?1
            ORDER BY split, recorded_at, dataset_snapshot_id
            "#,
        )
    }

    pub fn run_limit_records(&self, run_id: &str) -> Result<Vec<RunLimitsRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT record_json
            FROM run_limits
            WHERE run_id = ?1
            ORDER BY updated_at, run_limits_id
            "#,
        )
    }

    pub fn rendered_optimizer_state_records(
        &self,
        run_id: &str,
    ) -> Result<Vec<RenderedOptimizerStateRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT record_json
            FROM rendered_optimizer_states
            WHERE run_id = ?1
            ORDER BY sequence_number, rendered_state_id
            "#,
        )
    }

    pub fn runtime_effect_records(&self, run_id: &str) -> Result<Vec<RuntimeEffectRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT record_json
            FROM runtime_effects
            WHERE run_id = ?1
            ORDER BY planned_at, lane, runtime_effect_id
            "#,
        )
    }

    pub fn runtime_effect(
        &self,
        run_id: &str,
        runtime_effect_id: &str,
    ) -> Result<RuntimeEffectRecord> {
        self.store
            .conn
            .query_row(
                r#"
                SELECT record_json
                FROM runtime_effects
                WHERE run_id = ?1 AND runtime_effect_id = ?2
                "#,
                params![run_id, runtime_effect_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .map(|raw| serde_json::from_str(&raw).map_err(OptimizerError::from))
            .transpose()?
            .ok_or_else(|| {
                OptimizerError::Invariant(format!(
                    "runtime effect does not exist run_id={run_id} runtime_effect_id={runtime_effect_id}"
                ))
            })
    }

    pub fn runtime_effect_admission_records(
        &self,
        run_id: &str,
    ) -> Result<Vec<RuntimeEffectAdmissionRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT record_json
            FROM runtime_effect_admissions
            WHERE run_id = ?1
            ORDER BY checked_at, admission_id
            "#,
        )
    }

    pub fn optimizer_job_records(&self, run_id: &str) -> Result<Vec<OptimizerJob>> {
        let mut stmt = self.store.conn.prepare(
            r#"
            SELECT job_id, run_id, kind, status, candidate_id, attempt,
                   lease_id, worker_id, leased_at, lease_expires_at,
                   heartbeat_at, next_retry_at, retry_policy_json,
                   failure_json, payload_json
            FROM optimizer_jobs
            WHERE run_id = ?1
            ORDER BY updated_at, job_id
            "#,
        )?;
        let mut rows = stmt.query(params![run_id])?;
        let mut records = Vec::new();
        while let Some(row) = rows.next()? {
            records.push(optimizer_job_from_row(row)?);
        }
        Ok(records)
    }

    pub fn budget_reservation_records(&self, run_id: &str) -> Result<Vec<BudgetReservationRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT record_json
            FROM budget_reservations
            WHERE run_id = ?1
            ORDER BY reserved_at, budget_reservation_id
            "#,
        )
    }

    pub fn budget_reservation(
        &self,
        run_id: &str,
        budget_reservation_id: &str,
    ) -> Result<BudgetReservationRecord> {
        self.store
            .conn
            .query_row(
                r#"
                SELECT record_json
                FROM budget_reservations
                WHERE run_id = ?1 AND budget_reservation_id = ?2
                "#,
                params![run_id, budget_reservation_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .map(|raw| serde_json::from_str(&raw).map_err(OptimizerError::from))
            .transpose()?
            .ok_or_else(|| {
                OptimizerError::Invariant(format!(
                    "budget reservation does not exist run_id={run_id} budget_reservation_id={budget_reservation_id}"
                ))
            })
    }

    pub fn budget_commit_records(&self, run_id: &str) -> Result<Vec<BudgetCommitRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT record_json
            FROM budget_commits
            WHERE run_id = ?1
            ORDER BY committed_at, budget_commit_id
            "#,
        )
    }

    pub fn budget_release_records(&self, run_id: &str) -> Result<Vec<BudgetReleaseRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT record_json
            FROM budget_releases
            WHERE run_id = ?1
            ORDER BY released_at, budget_release_id
            "#,
        )
    }

    pub fn candidate_records(&self, run_id: &str) -> Result<Vec<Value>> {
        self.json_records(
            run_id,
            r#"
            SELECT record_json
            FROM candidates
            WHERE run_id = ?1
            ORDER BY updated_at, candidate_id
            "#,
        )
    }

    pub fn candidate_payload_records(&self, run_id: &str) -> Result<Vec<CandidatePayloadRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT payload_record_json
            FROM candidate_payloads
            WHERE run_id = ?1
            ORDER BY updated_at, candidate_payload_id
            "#,
        )
    }

    pub fn candidate_delta_records(&self, run_id: &str) -> Result<Vec<CandidateDeltaRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT delta_json
            FROM candidate_deltas
            WHERE run_id = ?1
            ORDER BY updated_at, candidate_delta_id
            "#,
        )
    }

    pub fn acceptance_decision_records(
        &self,
        run_id: &str,
    ) -> Result<Vec<AcceptanceDecisionRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT decision_json
            FROM acceptance_decisions
            WHERE run_id = ?1
            ORDER BY updated_at, acceptance_decision_id
            "#,
        )
    }

    pub fn frontier_cell_records(&self, run_id: &str) -> Result<Vec<FrontierCellRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT cell_json
            FROM frontier_cells
            WHERE run_id = ?1
            ORDER BY split, objective, rank, frontier_cell_id
            "#,
        )
    }

    pub fn plan_link_records(&self, run_id: &str) -> Result<Vec<PlanLinkRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT link_json
            FROM plan_links
            WHERE run_id = ?1
            ORDER BY relation, plan_link_id
            "#,
        )
    }

    pub fn materialization_records(&self, run_id: &str) -> Result<Vec<MaterializationRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT record_json
            FROM materializations
            WHERE run_id = ?1
            ORDER BY evaluation_stage, candidate_id, example_id, materialization_id
            "#,
        )
    }

    pub fn materialization_for_cache_key(
        &self,
        run_id: &str,
        cache_key: &str,
    ) -> Result<Option<MaterializationRecord>> {
        self.store
            .conn
            .query_row(
                r#"
                SELECT record_json
                FROM materializations
                WHERE run_id = ?1 AND cache_key = ?2
                ORDER BY updated_at DESC
                LIMIT 1
                "#,
                params![run_id, cache_key],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .map(|raw| serde_json::from_str(&raw).map_err(OptimizerError::from))
            .transpose()
    }

    pub fn sensor_frames(&self, run_id: &str) -> Result<Vec<SensorFrame>> {
        self.json_records(
            run_id,
            r#"
            SELECT frame_json
            FROM sensor_frames
            WHERE run_id = ?1
            ORDER BY evaluation_stage, candidate_id, example_id, sensor_frame_id
            "#,
        )
    }

    pub fn rollout_records(&self, run_id: &str) -> Result<Vec<RolloutRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT rollout_json
            FROM rollouts
            WHERE run_id = ?1
            ORDER BY evaluation_stage, candidate_id, example_id, rollout_record_id
            "#,
        )
    }

    pub fn rollout_event_records(&self, run_id: &str) -> Result<Vec<RolloutEventRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT event_record_json
            FROM rollout_events
            WHERE run_id = ?1
            ORDER BY rollout_record_id, sequence_number, rollout_event_id
            "#,
        )
    }

    pub fn rollout_event_records_for_rollout(
        &self,
        run_id: &str,
        rollout_id: &str,
    ) -> Result<Vec<RolloutEventRecord>> {
        let mut stmt = self.store.conn.prepare(
            r#"
            SELECT event.event_record_json
            FROM rollout_events event
            JOIN rollouts rollout
              ON rollout.run_id = event.run_id
             AND rollout.rollout_record_id = event.rollout_record_id
            WHERE event.run_id = ?1
              AND (
                    rollout.rollout_id = ?2
                 OR rollout.rollout_record_id = ?2
                 OR event.rollout_record_id = ?2
              )
            ORDER BY event.sequence_number, event.rollout_event_id
            "#,
        )?;
        let mut rows = stmt.query(params![run_id, rollout_id])?;
        let mut records = Vec::new();
        while let Some(row) = rows.next()? {
            let raw: String = row.get(0)?;
            records.push(serde_json::from_str(&raw)?);
        }
        Ok(records)
    }

    pub fn objective_set_records(&self, run_id: &str) -> Result<Vec<ObjectiveSetRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT objective_set_json
            FROM objective_sets
            WHERE run_id = ?1
            ORDER BY selection_objective, objective_set_id
            "#,
        )
    }

    pub fn objective_specs(&self, run_id: &str) -> Result<Vec<ObjectiveSpec>> {
        self.json_records(
            run_id,
            r#"
            SELECT objective_json
            FROM objectives
            WHERE run_id = ?1
            ORDER BY name, source, objective_id
            "#,
        )
    }

    pub fn score_records(&self, run_id: &str) -> Result<Vec<ScoreRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT score_json
            FROM scores
            WHERE run_id = ?1
            ORDER BY evaluation_stage, candidate_id, example_id, objective, score_id
            "#,
        )
    }

    pub fn score_vector_records(&self, run_id: &str) -> Result<Vec<ScoreVectorRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT score_vector_json
            FROM score_vectors
            WHERE run_id = ?1
            ORDER BY split, evaluation_stage, candidate_id, score_vector_id
            "#,
        )
    }

    pub fn pareto_comparison_records(&self, run_id: &str) -> Result<Vec<ParetoComparisonRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT comparison_json
            FROM pareto_comparisons
            WHERE run_id = ?1
            ORDER BY frontier_type, split, evaluation_stage, challenger_candidate_id, incumbent_candidate_id
            "#,
        )
    }

    pub fn trace_annotations(&self, run_id: &str) -> Result<Vec<TraceAnnotation>> {
        self.json_records(
            run_id,
            r#"
            SELECT annotation_json
            FROM trace_annotations
            WHERE run_id = ?1
            ORDER BY evaluation_stage, candidate_id, example_id, annotation_id
            "#,
        )
    }

    pub fn evidence_frames(&self, run_id: &str) -> Result<Vec<EvidenceFrame>> {
        self.json_records(
            run_id,
            r#"
            SELECT frame_json
            FROM evidence_frames
            WHERE run_id = ?1
            ORDER BY kind, subject_type, subject_id, evidence_frame_id
            "#,
        )
    }

    pub fn verifier_jobs(&self, run_id: &str) -> Result<Vec<VerifierJob>> {
        self.json_records(
            run_id,
            r#"
            SELECT job_json
            FROM verifier_jobs
            WHERE run_id = ?1
            ORDER BY status, verifier_id, verifier_job_id
            "#,
        )
    }

    pub fn subagent_invocations(&self, run_id: &str) -> Result<Vec<SubagentInvocation>> {
        self.json_records(
            run_id,
            r#"
            SELECT invocation_json
            FROM subagent_invocations
            WHERE run_id = ?1
            ORDER BY status, role, backend, invocation_id
            "#,
        )
    }

    pub fn cache_profile_records(&self, run_id: &str) -> Result<Vec<CacheProfileRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT profile_record_json
            FROM cache_profiles
            WHERE run_id = ?1
            ORDER BY cache_profile_id
            "#,
        )
    }

    pub fn cache_access_records(&self, run_id: &str) -> Result<Vec<CacheAccessRecord>> {
        self.json_records(
            run_id,
            r#"
            SELECT access_json
            FROM cache_accesses
            WHERE run_id = ?1
            ORDER BY sequence_number, cache_access_id
            "#,
        )
    }

    pub fn artifact_refs(&self, run_id: &str) -> Result<Vec<ArtifactRef>> {
        let mut stmt = self.store.conn.prepare(
            r#"
            SELECT path, kind, sha256, bytes, retention
            FROM artifact_refs
            WHERE run_id = ?1
            ORDER BY kind, path
            "#,
        )?;
        let mut rows = stmt.query(params![run_id])?;
        let mut refs = Vec::new();
        while let Some(row) = rows.next()? {
            let bytes = row.get::<_, i64>(3)?.max(0) as u64;
            refs.push(ArtifactRef {
                path: row.get(0)?,
                kind: row.get(1)?,
                sha256: row.get(2)?,
                bytes,
                retention: row.get(4)?,
            });
        }
        Ok(refs)
    }

    pub fn cache_summary(&self, run_id: &str) -> Result<Value> {
        let mut evaluation_stmt = self.store.conn.prepare(
            r#"
            SELECT cache_schema_version, cache_profile, status, COUNT(*) AS entries
            FROM evaluation_cache
            WHERE run_id = ?1
            GROUP BY cache_schema_version, cache_profile, status
            ORDER BY cache_schema_version, cache_profile, status
            "#,
        )?;
        let mut evaluation_rows = evaluation_stmt.query(params![run_id])?;
        let mut evaluation_cache = Vec::new();
        while let Some(row) = evaluation_rows.next()? {
            evaluation_cache.push(json!({
                "cache_schema_version": row.get::<_, String>(0)?,
                "cache_profile": row.get::<_, String>(1)?,
                "status": row.get::<_, String>(2)?,
                "entries": row.get::<_, i64>(3)?.max(0) as u64,
            }));
        }
        let mut access_stmt = self.store.conn.prepare(
            r#"
            SELECT namespace, boundary, action, status, COUNT(*) AS entries
            FROM cache_accesses
            WHERE run_id = ?1
            GROUP BY namespace, boundary, action, status
            ORDER BY namespace, boundary, action, status
            "#,
        )?;
        let mut access_rows = access_stmt.query(params![run_id])?;
        let mut cache_accesses = Vec::new();
        while let Some(row) = access_rows.next()? {
            cache_accesses.push(json!({
                "namespace": row.get::<_, String>(0)?,
                "boundary": row.get::<_, String>(1)?,
                "action": row.get::<_, String>(2)?,
                "status": row.get::<_, String>(3)?,
                "entries": row.get::<_, i64>(4)?.max(0) as u64,
            }));
        }
        Ok(json!({
            "evaluation_cache": evaluation_cache,
            "cache_accesses": cache_accesses,
        }))
    }

    pub fn evaluation_cache_entry(
        &self,
        run_id: &str,
        cache_key: &str,
    ) -> Result<Option<EvaluationCacheRecord>> {
        self.store
            .conn
            .query_row(
                r#"
                SELECT record_json
                FROM evaluation_cache
                WHERE run_id = ?1 AND cache_key = ?2
                "#,
                params![run_id, cache_key],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .map(|raw| serde_json::from_str(&raw).map_err(OptimizerError::from))
            .transpose()
    }

    pub fn evaluation_cache_entries(&self, run_id: &str) -> Result<Vec<EvaluationCacheRecord>> {
        let mut stmt = self.store.conn.prepare(
            r#"
            SELECT record_json
            FROM evaluation_cache
            WHERE run_id = ?1
            ORDER BY created_at, cache_key
            "#,
        )?;
        let mut rows = stmt.query(params![run_id])?;
        let mut records = Vec::new();
        while let Some(row) = rows.next()? {
            let raw: String = row.get(0)?;
            records.push(serde_json::from_str(&raw)?);
        }
        Ok(records)
    }
}

pub fn workspace_status(path: impl AsRef<Path>) -> Result<WorkspaceStatus> {
    WorkspaceStore::open_existing(path)?.status()
}

fn run_request_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<WorkspaceRunRequestStatus> {
    let usage_json = row.get::<_, Option<String>>(22)?;
    let result_json = row.get::<_, Option<String>>(23)?;
    let error_json = row.get::<_, Option<String>>(24)?;
    Ok(WorkspaceRunRequestStatus {
        request_id: row.get(0)?,
        run_id: row.get(1)?,
        status: row.get(2)?,
        config_path: row.get(3)?,
        container_url: row.get(4)?,
        cache_mode: row.get(5)?,
        cache_namespace: row.get(6)?,
        output_dir: row.get(7)?,
        run_dir: row.get(8)?,
        priority: row.get(9)?,
        submitted_at: row.get(10)?,
        leased_at: row.get(11)?,
        lease_expires_at: row.get(12)?,
        started_at: row.get(13)?,
        finished_at: row.get(14)?,
        updated_at: row.get(15)?,
        lease_id: row.get(16)?,
        worker_id: row.get(17)?,
        run_workspace_db_path: row.get(18)?,
        result_manifest_path: row.get(19)?,
        best_candidate_id: row.get(20)?,
        cost_usd: row.get(21)?,
        usage: parse_json_or_null(usage_json.as_deref()),
        result: parse_json_or_null(result_json.as_deref()),
        error: parse_json_or_null(error_json.as_deref()),
    })
}

fn optimizer_job_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<OptimizerJob> {
    let kind_text: String = row.get(2)?;
    let status_text: String = row.get(3)?;
    let retry_policy_json: String = row.get(12)?;
    let failure_json: Option<String> = row.get(13)?;
    let payload_json: String = row.get(14)?;
    let kind = OptimizerJobKind::parse(&kind_text).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            2,
            rusqlite::types::Type::Text,
            Box::new(OptimizerError::Invariant(format!(
                "unknown optimizer job kind {kind_text:?}"
            ))),
        )
    })?;
    let status = OptimizerJobStatus::parse(&status_text).ok_or_else(|| {
        rusqlite::Error::FromSqlConversionFailure(
            3,
            rusqlite::types::Type::Text,
            Box::new(OptimizerError::Invariant(format!(
                "unknown optimizer job status {status_text:?}"
            ))),
        )
    })?;
    let retry_policy = serde_json::from_str(&retry_policy_json).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(12, rusqlite::types::Type::Text, Box::new(error))
    })?;
    let failure = failure_json
        .as_deref()
        .map(serde_json::from_str)
        .transpose()
        .map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                13,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?;
    let payload = serde_json::from_str(&payload_json).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(14, rusqlite::types::Type::Text, Box::new(error))
    })?;
    Ok(OptimizerJob {
        job_id: row.get(0)?,
        run_id: row.get(1)?,
        kind,
        status,
        candidate_id: row.get(4)?,
        attempt: row.get::<_, i64>(5)?.max(0) as u32,
        lease_id: row.get(6)?,
        worker_id: row.get(7)?,
        leased_at: row.get(8)?,
        lease_expires_at: row.get(9)?,
        heartbeat_at: row.get(10)?,
        next_retry_at: row.get(11)?,
        retry_policy,
        failure,
        payload,
    })
}

#[derive(Clone, Debug, Default)]
struct CandidateSnapshot {
    payload: Value,
    lever_bundle: Value,
    minibatch_reward: Option<f64>,
    train_reward: Option<f64>,
}

#[derive(Clone, Debug)]
struct FrontierCandidate {
    candidate_id: String,
    parent_id: Option<String>,
    source: String,
    train_reward: f64,
    objective: String,
    score_vector: Value,
}

fn latest_objective_set_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
) -> Result<Option<ObjectiveSetRecord>> {
    tx.query_row(
        r#"
        SELECT objective_set_json
        FROM objective_sets
        WHERE run_id = ?1
        ORDER BY updated_at DESC, objective_set_id DESC
        LIMIT 1
        "#,
        params![run_id],
        |row| row.get::<_, String>(0),
    )
    .optional()?
    .map(|raw| serde_json::from_str(&raw).map_err(OptimizerError::from))
    .transpose()
}

fn rebuild_score_vectors_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
) -> Result<(Option<ObjectiveSetRecord>, Vec<ScoreVectorRecord>)> {
    tx.execute(
        "DELETE FROM pareto_comparisons WHERE run_id = ?1",
        params![run_id],
    )?;
    tx.execute(
        "DELETE FROM score_vectors WHERE run_id = ?1",
        params![run_id],
    )?;
    tx.execute(
        "DELETE FROM plan_links WHERE run_id = ?1 AND relation = 'score_vector'",
        params![run_id],
    )?;
    let Some(objective_set) = latest_objective_set_tx(tx, run_id)? else {
        return Ok((None, Vec::new()));
    };

    let mut stmt = tx.prepare(
        r#"
        SELECT score_json
        FROM scores
        WHERE run_id = ?1
        ORDER BY candidate_id, split, evaluation_stage, example_id, objective, score_id
        "#,
    )?;
    let mut rows = stmt.query(params![run_id])?;
    let mut grouped = BTreeMap::<(String, String, String), Vec<ScoreRecord>>::new();
    while let Some(row) = rows.next()? {
        let raw: String = row.get(0)?;
        let score: ScoreRecord = serde_json::from_str(&raw)?;
        grouped
            .entry((
                score.candidate_id.clone(),
                score.split.clone(),
                score.evaluation_stage.clone(),
            ))
            .or_default()
            .push(score);
    }
    drop(rows);
    drop(stmt);

    let mut vectors = Vec::new();
    for ((candidate_id, split, evaluation_stage), scores) in grouped {
        let mut metadata = Map::new();
        metadata.insert("source".to_string(), json!("workspace.scores"));
        metadata.insert("projection".to_string(), json!("score_vectors_from_scores"));
        let vector = ScoreVectorRecord::from_scores(
            &objective_set,
            &candidate_id,
            &split,
            &evaluation_stage,
            &scores,
            metadata,
        );
        upsert_score_vector_tx(tx, run_id, &vector)?;
        upsert_plan_link_tx(
            tx,
            run_id,
            &PlanLinkRecord::from_input(PlanLinkInput {
                source_type: "candidate",
                source_id: &candidate_id,
                target_type: "score_vector",
                target_id: &vector.score_vector_id,
                relation: "score_vector",
                status: "active",
                confidence: 1.0,
                metadata: Map::new(),
            }),
        )?;
        vectors.push(vector);
    }
    Ok((Some(objective_set), vectors))
}

fn rebuild_pareto_comparisons_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    objective_set: &ObjectiveSetRecord,
    score_vectors: &[ScoreVectorRecord],
) -> Result<()> {
    tx.execute(
        "DELETE FROM pareto_comparisons WHERE run_id = ?1",
        params![run_id],
    )?;
    let comparable = score_vectors
        .iter()
        .filter(|vector| {
            vector.split == "train"
                && vector.selection_score.is_some()
                && matches!(
                    vector.evaluation_stage.as_str(),
                    "seed_full_train" | "candidate_full_train"
                )
        })
        .collect::<Vec<_>>();
    for challenger in &comparable {
        for incumbent in &comparable {
            if challenger.candidate_id == incumbent.candidate_id {
                continue;
            }
            let mut metadata = Map::new();
            metadata.insert("source".to_string(), json!("workspace.score_vectors"));
            metadata.insert(
                "projection".to_string(),
                json!("pareto_comparisons_from_score_vectors"),
            );
            let comparison = ParetoComparisonRecord::from_vectors(
                objective_set,
                &objective_set.frontier_type,
                "train",
                "full_train",
                challenger,
                incumbent,
                metadata,
            );
            upsert_pareto_comparison_tx(tx, run_id, &comparison)?;
        }
    }
    Ok(())
}

fn candidate_snapshot_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    candidate_id: &str,
) -> Result<Option<CandidateSnapshot>> {
    tx.query_row(
        r#"
        SELECT payload_json, lever_bundle_json, minibatch_reward, train_reward
        FROM candidates
        WHERE run_id = ?1 AND candidate_id = ?2
        "#,
        params![run_id, candidate_id],
        |row| {
            let payload_json: String = row.get(0)?;
            let lever_bundle_json: String = row.get(1)?;
            Ok(CandidateSnapshot {
                payload: serde_json::from_str(&payload_json).unwrap_or(Value::Null),
                lever_bundle: serde_json::from_str(&lever_bundle_json).unwrap_or(Value::Null),
                minibatch_reward: row.get(2)?,
                train_reward: row.get(3)?,
            })
        },
    )
    .optional()
    .map_err(OptimizerError::from)
}

fn upsert_candidate_payload_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    record: &CandidatePayloadRecord,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO candidate_payloads(
            run_id, candidate_payload_id, candidate_id, parent_id, source,
            status, payload_hash, payload_json, lever_bundle_json, metadata_json,
            payload_record_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, datetime('now'))
        ON CONFLICT(run_id, candidate_payload_id) DO UPDATE SET
            candidate_id = excluded.candidate_id,
            parent_id = excluded.parent_id,
            source = excluded.source,
            status = excluded.status,
            payload_hash = excluded.payload_hash,
            payload_json = excluded.payload_json,
            lever_bundle_json = excluded.lever_bundle_json,
            metadata_json = excluded.metadata_json,
            payload_record_json = excluded.payload_record_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            record.candidate_payload_id,
            record.candidate_id,
            record.parent_id.as_deref(),
            record.source,
            record.status,
            record.payload_hash,
            stable_json(&record.payload),
            stable_json(&record.lever_bundle),
            stable_json(&serde_json::to_value(&record.metadata)?),
            stable_json(&serde_json::to_value(record)?),
        ],
    )?;
    Ok(())
}

fn upsert_candidate_delta_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    record: &CandidateDeltaRecord,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO candidate_deltas(
            run_id, candidate_delta_id, candidate_id, parent_candidate_id,
            operation_kind, source, status, target_levers_json,
            changed_fields_json, before_json, after_json, rationale,
            metadata_json, delta_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, datetime('now'))
        ON CONFLICT(run_id, candidate_delta_id) DO UPDATE SET
            candidate_id = excluded.candidate_id,
            parent_candidate_id = excluded.parent_candidate_id,
            operation_kind = excluded.operation_kind,
            source = excluded.source,
            status = excluded.status,
            target_levers_json = excluded.target_levers_json,
            changed_fields_json = excluded.changed_fields_json,
            before_json = excluded.before_json,
            after_json = excluded.after_json,
            rationale = excluded.rationale,
            metadata_json = excluded.metadata_json,
            delta_json = excluded.delta_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            record.candidate_delta_id,
            record.candidate_id,
            record.parent_candidate_id,
            record.operation_kind,
            record.source,
            record.status,
            stable_json(&serde_json::to_value(&record.target_levers)?),
            stable_json(&serde_json::to_value(&record.changed_fields)?),
            stable_json(&record.before),
            stable_json(&record.after),
            record.rationale,
            stable_json(&serde_json::to_value(&record.metadata)?),
            stable_json(&serde_json::to_value(record)?),
        ],
    )?;
    Ok(())
}

fn upsert_acceptance_decision_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    record: &AcceptanceDecisionRecord,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO acceptance_decisions(
            run_id, acceptance_decision_id, candidate_id, parent_candidate_id,
            decision, stage, status, reason, candidate_minibatch_reward,
            parent_minibatch_reward, candidate_train_reward, parent_train_reward,
            heldout_reward, score_json, metadata_json, decision_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, datetime('now'))
        ON CONFLICT(run_id, acceptance_decision_id) DO UPDATE SET
            candidate_id = excluded.candidate_id,
            parent_candidate_id = excluded.parent_candidate_id,
            decision = excluded.decision,
            stage = excluded.stage,
            status = excluded.status,
            reason = excluded.reason,
            candidate_minibatch_reward = excluded.candidate_minibatch_reward,
            parent_minibatch_reward = excluded.parent_minibatch_reward,
            candidate_train_reward = excluded.candidate_train_reward,
            parent_train_reward = excluded.parent_train_reward,
            heldout_reward = excluded.heldout_reward,
            score_json = excluded.score_json,
            metadata_json = excluded.metadata_json,
            decision_json = excluded.decision_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            record.acceptance_decision_id,
            record.candidate_id,
            record.parent_candidate_id.as_deref(),
            record.decision,
            record.stage,
            record.status,
            record.reason,
            record.candidate_minibatch_reward,
            record.parent_minibatch_reward,
            record.candidate_train_reward,
            record.parent_train_reward,
            record.heldout_reward,
            stable_json(&record.score),
            stable_json(&serde_json::to_value(&record.metadata)?),
            stable_json(&serde_json::to_value(record)?),
        ],
    )?;
    Ok(())
}

fn upsert_frontier_cell_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    record: &FrontierCellRecord,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO frontier_cells(
            run_id, frontier_cell_id, frontier_name, candidate_id,
            parent_candidate_id, source, status, split, objective, rank, score,
            score_vector_json, metadata_json, cell_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, datetime('now'))
        ON CONFLICT(run_id, frontier_cell_id) DO UPDATE SET
            frontier_name = excluded.frontier_name,
            candidate_id = excluded.candidate_id,
            parent_candidate_id = excluded.parent_candidate_id,
            source = excluded.source,
            status = excluded.status,
            split = excluded.split,
            objective = excluded.objective,
            rank = excluded.rank,
            score = excluded.score,
            score_vector_json = excluded.score_vector_json,
            metadata_json = excluded.metadata_json,
            cell_json = excluded.cell_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            record.frontier_cell_id,
            record.frontier_name,
            record.candidate_id,
            record.parent_candidate_id.as_deref(),
            record.source,
            record.status,
            record.split,
            record.objective,
            record.rank as i64,
            record.score,
            stable_json(&record.score_vector),
            stable_json(&serde_json::to_value(&record.metadata)?),
            stable_json(&serde_json::to_value(record)?),
        ],
    )?;
    Ok(())
}

fn upsert_plan_link_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    record: &PlanLinkRecord,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO plan_links(
            run_id, plan_link_id, source_type, source_id, target_type,
            target_id, relation, status, confidence, metadata_json,
            link_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, datetime('now'))
        ON CONFLICT(run_id, plan_link_id) DO UPDATE SET
            source_type = excluded.source_type,
            source_id = excluded.source_id,
            target_type = excluded.target_type,
            target_id = excluded.target_id,
            relation = excluded.relation,
            status = excluded.status,
            confidence = excluded.confidence,
            metadata_json = excluded.metadata_json,
            link_json = excluded.link_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            record.plan_link_id,
            record.source_type,
            record.source_id,
            record.target_type,
            record.target_id,
            record.relation,
            record.status,
            record.confidence,
            stable_json(&serde_json::to_value(&record.metadata)?),
            stable_json(&serde_json::to_value(record)?),
        ],
    )?;
    Ok(())
}

fn rebuild_frontier_cells_tx(tx: &rusqlite::Transaction<'_>, run_id: &str) -> Result<()> {
    tx.execute(
        "DELETE FROM frontier_cells WHERE run_id = ?1 AND frontier_name = 'train_reward_best'",
        params![run_id],
    )?;
    tx.execute(
        "DELETE FROM plan_links WHERE run_id = ?1 AND relation = 'frontier_membership'",
        params![run_id],
    )?;
    let mut stmt = tx.prepare(
        r#"
        SELECT c.candidate_id, c.parent_id, c.source, sv.selection_score,
               sv.selection_objective, sv.score_vector_json
        FROM score_vectors AS sv
        JOIN candidates AS c
          ON c.run_id = sv.run_id AND c.candidate_id = sv.candidate_id
        WHERE sv.run_id = ?1
          AND sv.split = 'train'
          AND sv.evaluation_stage IN ('seed_full_train', 'candidate_full_train')
          AND sv.selection_score IS NOT NULL
          AND sv.selection_score = (
            SELECT MAX(selection_score)
            FROM score_vectors
            WHERE run_id = ?1
              AND split = 'train'
              AND evaluation_stage IN ('seed_full_train', 'candidate_full_train')
              AND selection_score IS NOT NULL
          )
        ORDER BY c.candidate_id
        "#,
    )?;
    let mut rows = stmt.query(params![run_id])?;
    let mut frontier = Vec::new();
    while let Some(row) = rows.next()? {
        let score_vector_json: String = row.get(5)?;
        frontier.push(FrontierCandidate {
            candidate_id: row.get(0)?,
            parent_id: row.get(1)?,
            source: row.get(2)?,
            train_reward: row.get(3)?,
            objective: row.get(4)?,
            score_vector: serde_json::from_str(&score_vector_json)?,
        });
    }
    drop(rows);
    drop(stmt);
    for (index, candidate) in frontier.into_iter().enumerate() {
        let cell = FrontierCellRecord::from_input(FrontierCellInput {
            frontier_name: "train_reward_best",
            candidate_id: &candidate.candidate_id,
            parent_candidate_id: candidate.parent_id,
            source: &candidate.source,
            status: "active",
            split: "train",
            objective: &candidate.objective,
            rank: (index + 1) as u64,
            score: candidate.train_reward,
            score_vector: candidate.score_vector,
        });
        upsert_frontier_cell_tx(tx, run_id, &cell)?;
        upsert_plan_link_tx(
            tx,
            run_id,
            &PlanLinkRecord::from_input(PlanLinkInput {
                source_type: "candidate",
                source_id: &cell.candidate_id,
                target_type: "frontier_cell",
                target_id: &cell.frontier_cell_id,
                relation: "frontier_membership",
                status: "active",
                confidence: 1.0,
                metadata: Map::new(),
            }),
        )?;
    }
    Ok(())
}

fn upsert_cache_access_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    access: &CacheAccessRecord,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO cache_accesses(
            run_id, cache_access_id, sequence_number, mode, namespace,
            boundary, cache_key, action, status, request_hash, response_hash,
            metadata_json, access_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, datetime('now'))
        ON CONFLICT(run_id, cache_access_id) DO UPDATE SET
            sequence_number = excluded.sequence_number,
            mode = excluded.mode,
            namespace = excluded.namespace,
            boundary = excluded.boundary,
            cache_key = excluded.cache_key,
            action = excluded.action,
            status = excluded.status,
            request_hash = excluded.request_hash,
            response_hash = excluded.response_hash,
            metadata_json = excluded.metadata_json,
            access_json = excluded.access_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            access.cache_access_id,
            access.sequence_number as i64,
            access.mode,
            access.namespace,
            access.boundary,
            access.cache_key,
            access.action,
            access.status,
            access.request_hash.as_deref(),
            access.response_hash.as_deref(),
            stable_json(&serde_json::to_value(&access.metadata)?),
            stable_json(&serde_json::to_value(access)?),
        ],
    )?;
    Ok(())
}

fn upsert_event_stream_event_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    event: &EventStreamRecord,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO event_stream_events(
            run_id, event_id, sequence_number, event_type, message, timestamp,
            fields_json, event_json, event_record_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, datetime('now'))
        ON CONFLICT(run_id, event_id) DO UPDATE SET
            sequence_number = excluded.sequence_number,
            event_type = excluded.event_type,
            message = excluded.message,
            timestamp = excluded.timestamp,
            fields_json = excluded.fields_json,
            event_json = excluded.event_json,
            event_record_json = excluded.event_record_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            event.event_id,
            event.sequence_number as i64,
            event.event_type,
            event.message,
            event.timestamp,
            stable_json(&event.fields),
            stable_json(&event.event),
            stable_json(&serde_json::to_value(event)?),
        ],
    )?;
    Ok(())
}

fn upsert_usage_ledger_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    record: &UsageLedgerRecord,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO usage_ledger(
            run_id, usage_ledger_id, boundary, source_type, source_id,
            candidate_id, evaluation_stage, model, provider, call_count,
            prompt_tokens, completion_tokens, total_tokens, cost_usd,
            usage_json, metadata_json, ledger_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, datetime('now'))
        ON CONFLICT(run_id, usage_ledger_id) DO UPDATE SET
            boundary = excluded.boundary,
            source_type = excluded.source_type,
            source_id = excluded.source_id,
            candidate_id = excluded.candidate_id,
            evaluation_stage = excluded.evaluation_stage,
            model = excluded.model,
            provider = excluded.provider,
            call_count = excluded.call_count,
            prompt_tokens = excluded.prompt_tokens,
            completion_tokens = excluded.completion_tokens,
            total_tokens = excluded.total_tokens,
            cost_usd = excluded.cost_usd,
            usage_json = excluded.usage_json,
            metadata_json = excluded.metadata_json,
            ledger_json = excluded.ledger_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            record.usage_ledger_id,
            record.boundary,
            record.source_type,
            record.source_id,
            record.candidate_id.as_deref(),
            record.evaluation_stage.as_deref(),
            record.model.as_deref(),
            record.provider.as_deref(),
            record.call_count as i64,
            record.prompt_tokens as i64,
            record.completion_tokens as i64,
            record.total_tokens as i64,
            record.cost_usd,
            stable_json(&record.usage),
            stable_json(&serde_json::to_value(&record.metadata)?),
            stable_json(&serde_json::to_value(record)?),
        ],
    )?;
    Ok(())
}

fn upsert_stopper_state_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    record: &StopperStateRecord,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO stopper_states(
            run_id, stopper_state_id, sequence_number, status, reason,
            generation, candidate_id, evaluation_stage, rollout_count,
            max_total_rollouts, remaining_rollouts, cost_usd, max_cost_usd,
            cost_budget_enabled, budget_exhausted, checked_at, metadata_json,
            state_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, datetime('now'))
        ON CONFLICT(run_id, stopper_state_id) DO UPDATE SET
            sequence_number = excluded.sequence_number,
            status = excluded.status,
            reason = excluded.reason,
            generation = excluded.generation,
            candidate_id = excluded.candidate_id,
            evaluation_stage = excluded.evaluation_stage,
            rollout_count = excluded.rollout_count,
            max_total_rollouts = excluded.max_total_rollouts,
            remaining_rollouts = excluded.remaining_rollouts,
            cost_usd = excluded.cost_usd,
            max_cost_usd = excluded.max_cost_usd,
            cost_budget_enabled = excluded.cost_budget_enabled,
            budget_exhausted = excluded.budget_exhausted,
            checked_at = excluded.checked_at,
            metadata_json = excluded.metadata_json,
            state_json = excluded.state_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            record.stopper_state_id,
            record.sequence_number as i64,
            record.status,
            record.reason.as_deref(),
            record.generation.map(|generation| generation as i64),
            record.candidate_id.as_deref(),
            record.evaluation_stage.as_deref(),
            record.rollout_count as i64,
            record.max_total_rollouts as i64,
            record
                .remaining_rollouts
                .map(|remaining_rollouts| remaining_rollouts as i64),
            record.cost_usd,
            record.max_cost_usd,
            i64::from(record.cost_budget_enabled),
            i64::from(record.budget_exhausted),
            record.checked_at,
            stable_json(&serde_json::to_value(&record.metadata)?),
            stable_json(&serde_json::to_value(record)?),
        ],
    )?;
    Ok(())
}

fn upsert_checkpoint_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    record: &CheckpointRecord,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO checkpoints(
            run_id, checkpoint_id, sequence_number, checkpoint_kind, status,
            run_state, reason, generation, candidate_id, evaluation_stage,
            best_candidate_id, candidate_count, frontier_count, rollout_count,
            cost_usd, usage_json, snapshot_json, metadata_json, checkpoint_json,
            created_at, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, datetime('now'))
        ON CONFLICT(run_id, checkpoint_id) DO UPDATE SET
            sequence_number = excluded.sequence_number,
            checkpoint_kind = excluded.checkpoint_kind,
            status = excluded.status,
            run_state = excluded.run_state,
            reason = excluded.reason,
            generation = excluded.generation,
            candidate_id = excluded.candidate_id,
            evaluation_stage = excluded.evaluation_stage,
            best_candidate_id = excluded.best_candidate_id,
            candidate_count = excluded.candidate_count,
            frontier_count = excluded.frontier_count,
            rollout_count = excluded.rollout_count,
            cost_usd = excluded.cost_usd,
            usage_json = excluded.usage_json,
            snapshot_json = excluded.snapshot_json,
            metadata_json = excluded.metadata_json,
            checkpoint_json = excluded.checkpoint_json,
            created_at = excluded.created_at,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            record.checkpoint_id,
            record.sequence_number as i64,
            record.checkpoint_kind,
            record.status,
            record.run_state,
            record.reason.as_deref(),
            record.generation.map(|generation| generation as i64),
            record.candidate_id.as_deref(),
            record.evaluation_stage.as_deref(),
            record.best_candidate_id.as_deref(),
            record.candidate_count as i64,
            record.frontier_count as i64,
            record.rollout_count as i64,
            record.cost_usd,
            stable_json(&record.usage),
            stable_json(&record.snapshot),
            stable_json(&serde_json::to_value(&record.metadata)?),
            stable_json(&serde_json::to_value(record)?),
            record.created_at,
        ],
    )?;
    Ok(())
}

fn upsert_sensor_frame_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    frame: &SensorFrame,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO sensor_frames(
            run_id, sensor_frame_id, candidate_id, rollout_id, example_id,
            seed, split, evaluation_stage, reward, status, trace_digest_json,
            usage_json, failure_json, frame_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, datetime('now'))
        ON CONFLICT(run_id, sensor_frame_id) DO UPDATE SET
            candidate_id = excluded.candidate_id,
            rollout_id = excluded.rollout_id,
            example_id = excluded.example_id,
            seed = excluded.seed,
            split = excluded.split,
            evaluation_stage = excluded.evaluation_stage,
            reward = excluded.reward,
            status = excluded.status,
            trace_digest_json = excluded.trace_digest_json,
            usage_json = excluded.usage_json,
            failure_json = excluded.failure_json,
            frame_json = excluded.frame_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            frame.sensor_frame_id,
            frame.candidate_id,
            frame.rollout_id,
            frame.example_id,
            frame.seed,
            frame.split,
            frame.evaluation_stage,
            frame.reward,
            frame.status,
            frame
                .trace_digest
                .as_ref()
                .map(serde_json::to_value)
                .transpose()?
                .as_ref()
                .map(stable_json),
            stable_json(&frame.usage),
            frame
                .failure
                .as_ref()
                .map(serde_json::to_value)
                .transpose()?
                .as_ref()
                .map(stable_json),
            stable_json(&serde_json::to_value(frame)?),
        ],
    )?;
    Ok(())
}

fn upsert_rollout_job_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    frame: &SensorFrame,
) -> Result<()> {
    let job_id = format!(
        "rollout:{}:{}:{}",
        frame.candidate_id, frame.evaluation_stage, frame.sensor_frame_id
    );
    let status = if frame.failure.is_some() {
        "failed"
    } else {
        "completed"
    };
    tx.execute(
        r#"
        INSERT INTO rollout_jobs(
            run_id, job_id, candidate_id, sensor_frame_id, example_id,
            seed, split, evaluation_stage, status, reward, failure_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, datetime('now'))
        ON CONFLICT(run_id, job_id) DO UPDATE SET
            candidate_id = excluded.candidate_id,
            sensor_frame_id = excluded.sensor_frame_id,
            example_id = excluded.example_id,
            seed = excluded.seed,
            split = excluded.split,
            evaluation_stage = excluded.evaluation_stage,
            status = excluded.status,
            reward = excluded.reward,
            failure_json = excluded.failure_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            job_id,
            frame.candidate_id,
            frame.sensor_frame_id,
            frame.example_id,
            frame.seed,
            frame.split,
            frame.evaluation_stage,
            status,
            frame.reward,
            frame
                .failure
                .as_ref()
                .map(serde_json::to_value)
                .transpose()?
                .as_ref()
                .map(stable_json),
        ],
    )?;
    Ok(())
}

fn upsert_rollout_record_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    rollout: &RolloutRecord,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO rollouts(
            run_id, rollout_record_id, rollout_id, candidate_id, sensor_frame_id,
            example_id, seed, split, evaluation_stage, status, reward,
            trace_sha256, event_count, usage_json, failure_json, metadata_json,
            rollout_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, datetime('now'))
        ON CONFLICT(run_id, rollout_record_id) DO UPDATE SET
            rollout_id = excluded.rollout_id,
            candidate_id = excluded.candidate_id,
            sensor_frame_id = excluded.sensor_frame_id,
            example_id = excluded.example_id,
            seed = excluded.seed,
            split = excluded.split,
            evaluation_stage = excluded.evaluation_stage,
            status = excluded.status,
            reward = excluded.reward,
            trace_sha256 = excluded.trace_sha256,
            event_count = excluded.event_count,
            usage_json = excluded.usage_json,
            failure_json = excluded.failure_json,
            metadata_json = excluded.metadata_json,
            rollout_json = excluded.rollout_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            rollout.rollout_record_id,
            rollout.rollout_id,
            rollout.candidate_id,
            rollout.sensor_frame_id,
            rollout.example_id,
            rollout.seed,
            rollout.split,
            rollout.evaluation_stage,
            rollout.status,
            rollout.reward,
            rollout.trace_sha256,
            rollout.event_count as i64,
            stable_json(&rollout.usage),
            rollout
                .failure
                .as_ref()
                .map(serde_json::to_value)
                .transpose()?
                .as_ref()
                .map(stable_json),
            stable_json(&serde_json::to_value(&rollout.metadata)?),
            stable_json(&serde_json::to_value(rollout)?),
        ],
    )?;
    Ok(())
}

fn upsert_rollout_event_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    event: &RolloutEventRecord,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO rollout_events(
            run_id, rollout_event_id, rollout_record_id, candidate_id,
            sensor_frame_id, sequence_number, event_type, kind, summary,
            payload_json, trace_ref, event_json, event_record_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, datetime('now'))
        ON CONFLICT(run_id, rollout_event_id) DO UPDATE SET
            rollout_record_id = excluded.rollout_record_id,
            candidate_id = excluded.candidate_id,
            sensor_frame_id = excluded.sensor_frame_id,
            sequence_number = excluded.sequence_number,
            event_type = excluded.event_type,
            kind = excluded.kind,
            summary = excluded.summary,
            payload_json = excluded.payload_json,
            trace_ref = excluded.trace_ref,
            event_json = excluded.event_json,
            event_record_json = excluded.event_record_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            event.rollout_event_id,
            event.rollout_record_id,
            event.candidate_id,
            event.sensor_frame_id,
            event.sequence_number as i64,
            event.event_type,
            event.kind,
            event.summary,
            stable_json(&event.payload),
            event.trace_ref,
            stable_json(&event.event),
            stable_json(&serde_json::to_value(event)?),
        ],
    )?;
    Ok(())
}

fn upsert_objective_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    objective: &ObjectiveSpec,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO objectives(
            run_id, objective_id, name, direction, source, aggregation,
            split_policy, objective_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, datetime('now'))
        ON CONFLICT(run_id, objective_id) DO UPDATE SET
            name = excluded.name,
            direction = excluded.direction,
            source = excluded.source,
            aggregation = excluded.aggregation,
            split_policy = excluded.split_policy,
            objective_json = excluded.objective_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            objective.objective_id,
            objective.name,
            objective.direction,
            objective.source,
            objective.aggregation,
            objective.split_policy,
            stable_json(&serde_json::to_value(objective)?),
        ],
    )?;
    Ok(())
}

fn upsert_score_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    score: &ScoreRecord,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO scores(
            run_id, score_id, objective_id, objective, candidate_id,
            sensor_frame_id, rollout_id, example_id, seed, split,
            evaluation_stage, source, value, rationale, metadata_json,
            score_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, datetime('now'))
        ON CONFLICT(run_id, score_id) DO UPDATE SET
            objective_id = excluded.objective_id,
            objective = excluded.objective,
            candidate_id = excluded.candidate_id,
            sensor_frame_id = excluded.sensor_frame_id,
            rollout_id = excluded.rollout_id,
            example_id = excluded.example_id,
            seed = excluded.seed,
            split = excluded.split,
            evaluation_stage = excluded.evaluation_stage,
            source = excluded.source,
            value = excluded.value,
            rationale = excluded.rationale,
            metadata_json = excluded.metadata_json,
            score_json = excluded.score_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            score.score_id,
            score.objective_id,
            score.objective,
            score.candidate_id,
            score.sensor_frame_id,
            score.rollout_id,
            score.example_id,
            score.seed,
            score.split,
            score.evaluation_stage,
            score.source,
            score.value,
            score.rationale,
            stable_json(&serde_json::to_value(&score.metadata)?),
            stable_json(&serde_json::to_value(score)?),
        ],
    )?;
    Ok(())
}

struct OptimizerJobPersist<'a> {
    job_id: &'a str,
    kind: OptimizerJobKind,
    status: OptimizerJobStatus,
    candidate_id: Option<&'a str>,
    sensor_frame_id: Option<&'a str>,
    failure: Option<&'a crate::failures::FailurePayload>,
    payload: Value,
}

fn upsert_optimizer_job_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    job: &OptimizerJobPersist<'_>,
) -> Result<()> {
    let retry_policy = serde_json::to_value(RetryPolicy::default())?;
    tx.execute(
        r#"
        INSERT INTO optimizer_jobs(
            run_id, job_id, kind, status, candidate_id, sensor_frame_id,
            attempt, lease_id, retry_policy_json, failure_json, payload_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 1, NULL, ?7, ?8, ?9, datetime('now'))
        ON CONFLICT(run_id, job_id) DO UPDATE SET
            kind = excluded.kind,
            status = excluded.status,
            candidate_id = excluded.candidate_id,
            sensor_frame_id = excluded.sensor_frame_id,
            attempt = excluded.attempt,
            lease_id = excluded.lease_id,
            retry_policy_json = excluded.retry_policy_json,
            failure_json = excluded.failure_json,
            payload_json = excluded.payload_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            job.job_id,
            job.kind.as_str(),
            job.status.as_str(),
            job.candidate_id,
            job.sensor_frame_id,
            stable_json(&retry_policy),
            job.failure
                .map(serde_json::to_value)
                .transpose()?
                .as_ref()
                .map(stable_json),
            stable_json(&job.payload),
        ],
    )?;
    Ok(())
}

fn upsert_score_vector_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    record: &ScoreVectorRecord,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO score_vectors(
            run_id, score_vector_id, objective_set_id, objective_set_hash,
            candidate_id, split, evaluation_stage, status, selection_objective,
            selection_score, mean_reward, score_count, objective_values_json,
            covered_objectives_json, missing_objectives_json, example_ids_json,
            seeds_json, metadata_json, score_vector_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, datetime('now'))
        ON CONFLICT(run_id, score_vector_id) DO UPDATE SET
            objective_set_id = excluded.objective_set_id,
            objective_set_hash = excluded.objective_set_hash,
            candidate_id = excluded.candidate_id,
            split = excluded.split,
            evaluation_stage = excluded.evaluation_stage,
            status = excluded.status,
            selection_objective = excluded.selection_objective,
            selection_score = excluded.selection_score,
            mean_reward = excluded.mean_reward,
            score_count = excluded.score_count,
            objective_values_json = excluded.objective_values_json,
            covered_objectives_json = excluded.covered_objectives_json,
            missing_objectives_json = excluded.missing_objectives_json,
            example_ids_json = excluded.example_ids_json,
            seeds_json = excluded.seeds_json,
            metadata_json = excluded.metadata_json,
            score_vector_json = excluded.score_vector_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            record.score_vector_id,
            record.objective_set_id,
            record.objective_set_hash,
            record.candidate_id,
            record.split,
            record.evaluation_stage,
            record.status,
            record.selection_objective,
            record.selection_score,
            record.mean_reward,
            record.score_count as i64,
            stable_json(&serde_json::to_value(&record.objective_values)?),
            stable_json(&serde_json::to_value(&record.covered_objectives)?),
            stable_json(&serde_json::to_value(&record.missing_objectives)?),
            stable_json(&serde_json::to_value(&record.example_ids)?),
            stable_json(&serde_json::to_value(&record.seeds)?),
            stable_json(&serde_json::to_value(&record.metadata)?),
            stable_json(&serde_json::to_value(record)?),
        ],
    )?;
    Ok(())
}

fn upsert_pareto_comparison_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    record: &ParetoComparisonRecord,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO pareto_comparisons(
            run_id, pareto_comparison_id, objective_set_id, objective_set_hash,
            frontier_type, split, evaluation_stage, challenger_candidate_id,
            incumbent_candidate_id, challenger_score_vector_id,
            incumbent_score_vector_id, result, dominance_json, rationale,
            metadata_json, comparison_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, datetime('now'))
        ON CONFLICT(run_id, pareto_comparison_id) DO UPDATE SET
            objective_set_id = excluded.objective_set_id,
            objective_set_hash = excluded.objective_set_hash,
            frontier_type = excluded.frontier_type,
            split = excluded.split,
            evaluation_stage = excluded.evaluation_stage,
            challenger_candidate_id = excluded.challenger_candidate_id,
            incumbent_candidate_id = excluded.incumbent_candidate_id,
            challenger_score_vector_id = excluded.challenger_score_vector_id,
            incumbent_score_vector_id = excluded.incumbent_score_vector_id,
            result = excluded.result,
            dominance_json = excluded.dominance_json,
            rationale = excluded.rationale,
            metadata_json = excluded.metadata_json,
            comparison_json = excluded.comparison_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            record.pareto_comparison_id,
            record.objective_set_id,
            record.objective_set_hash,
            record.frontier_type,
            record.split,
            record.evaluation_stage,
            record.challenger_candidate_id,
            record.incumbent_candidate_id,
            record.challenger_score_vector_id,
            record.incumbent_score_vector_id,
            record.result,
            stable_json(&record.dominance),
            record.rationale,
            stable_json(&serde_json::to_value(&record.metadata)?),
            stable_json(&serde_json::to_value(record)?),
        ],
    )?;
    Ok(())
}

fn upsert_trace_annotation_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    annotation: &TraceAnnotation,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO trace_annotations(
            run_id, annotation_id, sensor_frame_id, candidate_id, rollout_id,
            example_id, evaluation_stage, backend, status, summary, trace_sha256,
            event_count, llm_request_count, tool_call_count, call_site_ids_json,
            support_count, confidence, annotation_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, datetime('now'))
        ON CONFLICT(run_id, annotation_id) DO UPDATE SET
            sensor_frame_id = excluded.sensor_frame_id,
            candidate_id = excluded.candidate_id,
            rollout_id = excluded.rollout_id,
            example_id = excluded.example_id,
            evaluation_stage = excluded.evaluation_stage,
            backend = excluded.backend,
            status = excluded.status,
            summary = excluded.summary,
            trace_sha256 = excluded.trace_sha256,
            event_count = excluded.event_count,
            llm_request_count = excluded.llm_request_count,
            tool_call_count = excluded.tool_call_count,
            call_site_ids_json = excluded.call_site_ids_json,
            support_count = excluded.support_count,
            confidence = excluded.confidence,
            annotation_json = excluded.annotation_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            annotation.annotation_id,
            annotation.sensor_frame_id,
            annotation.candidate_id,
            annotation.rollout_id,
            annotation.example_id,
            annotation.evaluation_stage,
            annotation.backend,
            annotation.status,
            annotation.summary,
            annotation.trace_sha256,
            annotation.event_count as i64,
            annotation.llm_request_count as i64,
            annotation.tool_call_count as i64,
            stable_json(&serde_json::to_value(&annotation.call_site_ids)?),
            annotation.support_count as i64,
            annotation.confidence,
            stable_json(&serde_json::to_value(annotation)?),
        ],
    )?;
    Ok(())
}

fn upsert_evidence_frame_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    frame: &EvidenceFrame,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO evidence_frames(
            run_id, evidence_frame_id, subject_type, subject_id, candidate_id,
            sensor_frame_id, kind, source, summary, score, severity,
            evidence_json, frame_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, datetime('now'))
        ON CONFLICT(run_id, evidence_frame_id) DO UPDATE SET
            subject_type = excluded.subject_type,
            subject_id = excluded.subject_id,
            candidate_id = excluded.candidate_id,
            sensor_frame_id = excluded.sensor_frame_id,
            kind = excluded.kind,
            source = excluded.source,
            summary = excluded.summary,
            score = excluded.score,
            severity = excluded.severity,
            evidence_json = excluded.evidence_json,
            frame_json = excluded.frame_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            frame.evidence_frame_id,
            frame.subject_type,
            frame.subject_id,
            frame.candidate_id,
            frame.sensor_frame_id,
            frame.kind,
            frame.source,
            frame.summary,
            frame.score,
            frame.severity,
            stable_json(&frame.evidence),
            stable_json(&serde_json::to_value(frame)?),
        ],
    )?;
    Ok(())
}

fn upsert_verifier_job_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    job: &VerifierJob,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO verifier_jobs(
            run_id, verifier_job_id, verifier_id, candidate_id, sensor_frame_id,
            target_type, target_id, status, score, severity, summary, result_json,
            failure_json, evidence_frame_ids_json, job_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, datetime('now'))
        ON CONFLICT(run_id, verifier_job_id) DO UPDATE SET
            verifier_id = excluded.verifier_id,
            candidate_id = excluded.candidate_id,
            sensor_frame_id = excluded.sensor_frame_id,
            target_type = excluded.target_type,
            target_id = excluded.target_id,
            status = excluded.status,
            score = excluded.score,
            severity = excluded.severity,
            summary = excluded.summary,
            result_json = excluded.result_json,
            failure_json = excluded.failure_json,
            evidence_frame_ids_json = excluded.evidence_frame_ids_json,
            job_json = excluded.job_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            job.verifier_job_id,
            job.verifier_id,
            job.candidate_id,
            job.sensor_frame_id,
            job.target_type,
            job.target_id,
            job.status,
            job.score,
            job.severity,
            job.summary,
            stable_json(&job.result),
            job.failure
                .as_ref()
                .map(serde_json::to_value)
                .transpose()?
                .as_ref()
                .map(stable_json),
            stable_json(&serde_json::to_value(&job.evidence_frame_ids)?),
            stable_json(&serde_json::to_value(job)?),
        ],
    )?;
    Ok(())
}

fn upsert_subagent_invocation_tx(
    tx: &rusqlite::Transaction<'_>,
    run_id: &str,
    invocation: &SubagentInvocation,
) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO subagent_invocations(
            run_id, invocation_id, role, backend, trigger, candidate_id,
            sensor_frame_id, target_type, target_id, status, input_json,
            result_json, usage_json, cost_usd, failure_json, invocation_json,
            updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, datetime('now'))
        ON CONFLICT(run_id, invocation_id) DO UPDATE SET
            role = excluded.role,
            backend = excluded.backend,
            trigger = excluded.trigger,
            candidate_id = excluded.candidate_id,
            sensor_frame_id = excluded.sensor_frame_id,
            target_type = excluded.target_type,
            target_id = excluded.target_id,
            status = excluded.status,
            input_json = excluded.input_json,
            result_json = excluded.result_json,
            usage_json = excluded.usage_json,
            cost_usd = excluded.cost_usd,
            failure_json = excluded.failure_json,
            invocation_json = excluded.invocation_json,
            updated_at = datetime('now')
        "#,
        params![
            run_id,
            invocation.invocation_id,
            invocation.role,
            invocation.backend,
            invocation.trigger,
            invocation.candidate_id,
            invocation.sensor_frame_id,
            invocation.target_type,
            invocation.target_id,
            invocation.status,
            stable_json(&invocation.input),
            stable_json(&serde_json::to_value(&invocation.result)?),
            stable_json(&invocation.usage),
            invocation.cost_usd,
            invocation
                .failure
                .as_ref()
                .map(serde_json::to_value)
                .transpose()?
                .as_ref()
                .map(stable_json),
            stable_json(&serde_json::to_value(invocation)?),
        ],
    )?;
    Ok(())
}

fn run_projection_freshness(
    run_id: &str,
    state: &str,
    counts: &RunHealthCounts,
    checked_at: &str,
) -> Vec<ProjectionFreshnessRecord> {
    let mut records = vec![
        ProjectionFreshnessRecord::derived_count(
            run_id,
            "resolved_config_from_run_start",
            "optimization_runs",
            1,
            "resolved_run_configs",
            counts.resolved_run_configs,
            checked_at,
        ),
        ProjectionFreshnessRecord::derived_count(
            run_id,
            "run_limits_from_run_start",
            "optimization_runs",
            1,
            "run_limits",
            counts.run_limits,
            checked_at,
        ),
        ProjectionFreshnessRecord::exact_count(
            run_id,
            "runtime_effect_admissions_from_runtime_effects",
            "runtime_effects",
            counts.runtime_effects,
            "runtime_effect_admissions",
            counts.runtime_effect_admissions,
            checked_at,
        ),
        ProjectionFreshnessRecord::exact_count(
            run_id,
            "budget_releases_from_budget_commits",
            "budget_commits",
            counts.budget_commits,
            "budget_releases",
            counts.budget_releases,
            checked_at,
        ),
        ProjectionFreshnessRecord::exact_count(
            run_id,
            "candidate_payloads_from_candidates",
            "candidates",
            counts.candidates,
            "candidate_payloads",
            counts.candidate_payloads,
            checked_at,
        ),
        ProjectionFreshnessRecord::exact_count(
            run_id,
            "candidate_deltas_from_parented_candidates",
            "candidates.parent_id",
            counts.parented_candidates,
            "candidate_deltas",
            counts.candidate_deltas,
            checked_at,
        ),
        ProjectionFreshnessRecord::exact_count(
            run_id,
            "acceptance_decisions_from_candidates",
            "candidates",
            counts.candidates,
            "acceptance_decisions",
            counts.acceptance_decisions,
            checked_at,
        ),
        ProjectionFreshnessRecord::derived_count(
            run_id,
            "plan_links_from_candidates",
            "candidates",
            counts.candidates,
            "plan_links",
            counts.plan_links,
            checked_at,
        ),
        ProjectionFreshnessRecord::exact_count(
            run_id,
            "cache_accesses_from_cache_profile",
            "cache_profile.total_accesses",
            counts.cache_profile_access_total,
            "cache_accesses",
            counts.cache_accesses,
            checked_at,
        ),
        ProjectionFreshnessRecord::exact_count(
            run_id,
            "materializations_from_sensor_frames",
            "sensor_frames",
            counts.sensor_frames,
            "materializations",
            counts.materializations,
            checked_at,
        ),
        ProjectionFreshnessRecord::exact_count(
            run_id,
            "evaluation_cache_from_materialization_cache_keys",
            "materializations.distinct_cache_key",
            counts.evaluation_cache_expected,
            "evaluation_cache",
            counts.evaluation_cache,
            checked_at,
        ),
        ProjectionFreshnessRecord::derived_count(
            run_id,
            "event_stream_events_cover_state_transitions",
            "optimizer_state_history",
            counts.state_transitions,
            "event_stream_events",
            counts.event_stream_events,
            checked_at,
        ),
        ProjectionFreshnessRecord::exact_count(
            run_id,
            "rendered_states_from_state_transitions",
            "optimizer_state_history",
            counts.state_transitions,
            "rendered_optimizer_states",
            counts.rendered_optimizer_states,
            checked_at,
        ),
        ProjectionFreshnessRecord::exact_count(
            run_id,
            "usage_ledger_from_external_calls",
            "sensor_frames_plus_proposer_events",
            counts.usage_ledger_expected,
            "usage_ledger",
            counts.usage_ledger,
            checked_at,
        ),
        ProjectionFreshnessRecord::exact_count(
            run_id,
            "frontier_cells_from_train_frontier",
            "candidates.train_reward.max",
            counts.train_frontier_candidates,
            "frontier_cells",
            counts.frontier_cells,
            checked_at,
        ),
        ProjectionFreshnessRecord::exact_count(
            run_id,
            "rollout_jobs_from_sensor_frames",
            "sensor_frames",
            counts.sensor_frames,
            "rollout_jobs",
            counts.rollout_jobs,
            checked_at,
        ),
        ProjectionFreshnessRecord::exact_count(
            run_id,
            "rollouts_from_sensor_frames",
            "sensor_frames",
            counts.sensor_frames,
            "rollouts",
            counts.rollouts,
            checked_at,
        ),
        ProjectionFreshnessRecord::derived_count(
            run_id,
            "rollout_events_from_rollouts",
            "rollouts",
            counts.rollouts,
            "rollout_events",
            counts.rollout_events,
            checked_at,
        ),
        ProjectionFreshnessRecord::derived_count(
            run_id,
            "scores_from_sensor_frames",
            "sensor_frames",
            counts.sensor_frames,
            "scores",
            counts.scores,
            checked_at,
        ),
        ProjectionFreshnessRecord::exact_count(
            run_id,
            "score_vectors_from_scores",
            "distinct_score_candidate_split_stage",
            counts.score_vector_sources,
            "score_vectors",
            counts.score_vectors,
            checked_at,
        ),
        ProjectionFreshnessRecord::derived_count(
            run_id,
            "pareto_comparisons_from_score_vectors",
            "score_vectors",
            if counts.score_vectors > 1 {
                counts.score_vectors
            } else {
                0
            },
            "pareto_comparisons",
            counts.pareto_comparisons,
            checked_at,
        ),
        ProjectionFreshnessRecord::exact_count(
            run_id,
            "trace_annotations_from_sensor_frames",
            "sensor_frames",
            counts.sensor_frames,
            "trace_annotations",
            counts.trace_annotations,
            checked_at,
        ),
        ProjectionFreshnessRecord::derived_count(
            run_id,
            "evidence_frames_from_sensor_frames",
            "sensor_frames",
            counts.sensor_frames,
            "evidence_frames",
            counts.evidence_frames,
            checked_at,
        ),
        ProjectionFreshnessRecord::exact_count(
            run_id,
            "verifier_jobs_from_sensor_frames",
            "sensor_frames",
            counts.sensor_frames,
            "verifier_jobs",
            counts.verifier_jobs,
            checked_at,
        ),
        ProjectionFreshnessRecord::exact_count(
            run_id,
            "subagent_invocations_from_sensor_frames",
            "sensor_frames",
            counts.sensor_frames,
            "subagent_invocations",
            counts.subagent_invocations,
            checked_at,
        ),
    ];
    if is_terminal_run_state(state) {
        records.push(ProjectionFreshnessRecord::derived_count(
            run_id,
            "contract_snapshot_from_terminal_run",
            "optimization_runs.terminal",
            1,
            "container_contract_snapshots",
            counts.container_contract_snapshots,
            checked_at,
        ));
        records.push(ProjectionFreshnessRecord::derived_count(
            run_id,
            "prompt_program_snapshot_from_terminal_run",
            "optimization_runs.terminal",
            1,
            "prompt_program_snapshots",
            counts.prompt_program_snapshots,
            checked_at,
        ));
        records.push(ProjectionFreshnessRecord::derived_count(
            run_id,
            "dataset_snapshots_from_terminal_run",
            "optimization_runs.terminal",
            2,
            "dataset_snapshots",
            counts.dataset_snapshots,
            checked_at,
        ));
        records.push(ProjectionFreshnessRecord::derived_count(
            run_id,
            "objective_set_from_terminal_run",
            "optimization_runs.terminal",
            1,
            "objective_sets",
            counts.objective_sets,
            checked_at,
        ));
        records.push(ProjectionFreshnessRecord::exact_count(
            run_id,
            "cache_profile_from_terminal_run",
            "optimization_runs.terminal",
            1,
            "cache_profiles",
            counts.cache_profiles,
            checked_at,
        ));
        records.push(ProjectionFreshnessRecord::derived_count(
            run_id,
            "stopper_state_from_terminal_run",
            "optimization_runs.terminal",
            1,
            "stopper_states",
            counts.stopper_states,
            checked_at,
        ));
        records.push(ProjectionFreshnessRecord::derived_count(
            run_id,
            "checkpoint_from_terminal_run",
            "optimization_runs.terminal",
            1,
            "checkpoints",
            counts.checkpoints,
            checked_at,
        ));
        records.push(ProjectionFreshnessRecord::derived_count(
            run_id,
            "manifest_from_terminal_run",
            "optimization_runs.terminal",
            1,
            "manifests",
            counts.manifests,
            checked_at,
        ));
    }
    records
}

fn build_run_invariant_report(
    run_id: &str,
    state: &str,
    best_candidate_id: Option<String>,
    counts: &RunHealthCounts,
    checked_at: &str,
) -> InvariantReport {
    let mut violations = Vec::new();
    push_at_least_count_violation(
        &mut violations,
        run_id,
        "run_has_resolved_config",
        "optimization_runs",
        1,
        "resolved_run_configs",
        counts.resolved_run_configs,
    );
    push_at_least_count_violation(
        &mut violations,
        run_id,
        "run_has_run_limits",
        "optimization_runs",
        1,
        "run_limits",
        counts.run_limits,
    );
    push_exact_count_violation(
        &mut violations,
        run_id,
        "runtime_effect_admissions_match_runtime_effects",
        "runtime_effects",
        counts.runtime_effects,
        "runtime_effect_admissions",
        counts.runtime_effect_admissions,
    );
    push_exact_count_violation(
        &mut violations,
        run_id,
        "budget_releases_match_budget_commits",
        "budget_commits",
        counts.budget_commits,
        "budget_releases",
        counts.budget_releases,
    );
    push_exact_count_violation(
        &mut violations,
        run_id,
        "candidate_payloads_match_candidates",
        "candidates",
        counts.candidates,
        "candidate_payloads",
        counts.candidate_payloads,
    );
    push_exact_count_violation(
        &mut violations,
        run_id,
        "candidate_deltas_match_parented_candidates",
        "candidates.parent_id",
        counts.parented_candidates,
        "candidate_deltas",
        counts.candidate_deltas,
    );
    push_exact_count_violation(
        &mut violations,
        run_id,
        "acceptance_decisions_match_candidates",
        "candidates",
        counts.candidates,
        "acceptance_decisions",
        counts.acceptance_decisions,
    );
    push_at_least_count_violation(
        &mut violations,
        run_id,
        "plan_links_cover_candidates",
        "candidates",
        counts.candidates,
        "plan_links",
        counts.plan_links,
    );
    push_exact_count_violation(
        &mut violations,
        run_id,
        "cache_accesses_match_cache_profile_total",
        "cache_profile.total_accesses",
        counts.cache_profile_access_total,
        "cache_accesses",
        counts.cache_accesses,
    );
    push_exact_count_violation(
        &mut violations,
        run_id,
        "materializations_match_sensor_frames",
        "sensor_frames",
        counts.sensor_frames,
        "materializations",
        counts.materializations,
    );
    push_exact_count_violation(
        &mut violations,
        run_id,
        "evaluation_cache_matches_materialization_cache_keys",
        "materializations.distinct_cache_key",
        counts.evaluation_cache_expected,
        "evaluation_cache",
        counts.evaluation_cache,
    );
    push_at_least_count_violation(
        &mut violations,
        run_id,
        "event_stream_events_cover_state_transitions",
        "optimizer_state_history",
        counts.state_transitions,
        "event_stream_events",
        counts.event_stream_events,
    );
    push_exact_count_violation(
        &mut violations,
        run_id,
        "rendered_states_match_state_transitions",
        "optimizer_state_history",
        counts.state_transitions,
        "rendered_optimizer_states",
        counts.rendered_optimizer_states,
    );
    push_exact_count_violation(
        &mut violations,
        run_id,
        "usage_ledger_matches_external_calls",
        "sensor_frames_plus_proposer_events",
        counts.usage_ledger_expected,
        "usage_ledger",
        counts.usage_ledger,
    );
    push_exact_count_violation(
        &mut violations,
        run_id,
        "frontier_cells_match_train_frontier",
        "candidates.train_reward.max",
        counts.train_frontier_candidates,
        "frontier_cells",
        counts.frontier_cells,
    );
    push_exact_count_violation(
        &mut violations,
        run_id,
        "rollout_jobs_match_sensor_frames",
        "sensor_frames",
        counts.sensor_frames,
        "rollout_jobs",
        counts.rollout_jobs,
    );
    push_exact_count_violation(
        &mut violations,
        run_id,
        "rollouts_match_sensor_frames",
        "sensor_frames",
        counts.sensor_frames,
        "rollouts",
        counts.rollouts,
    );
    push_at_least_count_violation(
        &mut violations,
        run_id,
        "rollout_events_cover_rollouts",
        "rollouts",
        counts.rollouts,
        "rollout_events",
        counts.rollout_events,
    );
    push_at_least_count_violation(
        &mut violations,
        run_id,
        "scores_cover_sensor_frames",
        "sensor_frames",
        counts.sensor_frames,
        "scores",
        counts.scores,
    );
    push_exact_count_violation(
        &mut violations,
        run_id,
        "score_vectors_match_score_groups",
        "distinct_score_candidate_split_stage",
        counts.score_vector_sources,
        "score_vectors",
        counts.score_vectors,
    );
    push_exact_count_violation(
        &mut violations,
        run_id,
        "trace_annotations_match_sensor_frames",
        "sensor_frames",
        counts.sensor_frames,
        "trace_annotations",
        counts.trace_annotations,
    );
    push_at_least_count_violation(
        &mut violations,
        run_id,
        "evidence_frames_cover_sensor_frames",
        "sensor_frames",
        counts.sensor_frames,
        "evidence_frames",
        counts.evidence_frames,
    );
    push_exact_count_violation(
        &mut violations,
        run_id,
        "verifier_jobs_match_sensor_frames",
        "sensor_frames",
        counts.sensor_frames,
        "verifier_jobs",
        counts.verifier_jobs,
    );
    push_exact_count_violation(
        &mut violations,
        run_id,
        "subagent_invocations_match_sensor_frames",
        "sensor_frames",
        counts.sensor_frames,
        "subagent_invocations",
        counts.subagent_invocations,
    );
    if counts.scores > 0 && counts.objectives == 0 {
        violations.push(InvariantViolation::new(InvariantViolationInput {
            run_id,
            invariant_id: "scores_have_objectives",
            severity: "error",
            subject_type: "scores",
            subject_id: "objectives",
            message: "score rows require at least one objective row".to_string(),
            repair_hint: Some("rebuild objective rows from sensor frames".to_string()),
            details: json!({
                "scores": counts.scores,
                "objectives": counts.objectives,
            }),
        }));
    }
    if counts.scores > 0 && counts.objective_sets == 0 {
        violations.push(InvariantViolation::new(InvariantViolationInput {
            run_id,
            invariant_id: "scores_have_objective_set",
            severity: "error",
            subject_type: "scores",
            subject_id: "objective_sets",
            message: "score rows require a declared objective set row".to_string(),
            repair_hint: Some(
                "rebuild the run objective set from the prompt program and objective config"
                    .to_string(),
            ),
            details: json!({
                "scores": counts.scores,
                "objective_sets": counts.objective_sets,
            }),
        }));
    }
    if is_terminal_run_state(state) {
        if counts.container_contract_snapshots == 0 {
            violations.push(InvariantViolation::new(InvariantViolationInput {
                run_id,
                invariant_id: "terminal_run_has_container_contract_snapshot",
                severity: "error",
                subject_type: "optimization_runs",
                subject_id: run_id,
                message: "terminal runs must record the validated container contract snapshot"
                    .to_string(),
                repair_hint: Some(
                    "rebuild the contract snapshot from /metadata evidence".to_string(),
                ),
                details: json!({
                    "state": state,
                    "container_contract_snapshots": counts.container_contract_snapshots,
                }),
            }));
        }
        if counts.prompt_program_snapshots == 0 {
            violations.push(InvariantViolation::new(InvariantViolationInput {
                run_id,
                invariant_id: "terminal_run_has_prompt_program_snapshot",
                severity: "error",
                subject_type: "optimization_runs",
                subject_id: run_id,
                message: "terminal runs must record the prompt program snapshot".to_string(),
                repair_hint: Some("rebuild the prompt program snapshot from /program".to_string()),
                details: json!({
                    "state": state,
                    "prompt_program_snapshots": counts.prompt_program_snapshots,
                }),
            }));
        }
        if counts.dataset_snapshots < 2 {
            violations.push(InvariantViolation::new(InvariantViolationInput {
                run_id,
                invariant_id: "terminal_run_has_dataset_snapshots",
                severity: "error",
                subject_type: "optimization_runs",
                subject_id: run_id,
                message: "terminal runs must record train and heldout dataset snapshots"
                    .to_string(),
                repair_hint: Some(
                    "rebuild dataset snapshots from requested splits, seeds, and rows".to_string(),
                ),
                details: json!({"state": state, "dataset_snapshots": counts.dataset_snapshots}),
            }));
        }
        if counts.objective_sets == 0 {
            violations.push(InvariantViolation::new(InvariantViolationInput {
                run_id,
                invariant_id: "terminal_run_has_objective_set",
                severity: "error",
                subject_type: "optimization_runs",
                subject_id: run_id,
                message: "terminal runs must declare the objective set used for evaluation"
                    .to_string(),
                repair_hint: Some(
                    "rebuild the objective set from the prompt program and run config".to_string(),
                ),
                details: json!({"state": state, "objective_sets": counts.objective_sets}),
            }));
        }
        if counts.cache_profiles == 0 {
            violations.push(InvariantViolation::new(InvariantViolationInput {
                run_id,
                invariant_id: "terminal_run_has_cache_profile",
                severity: "error",
                subject_type: "optimization_runs",
                subject_id: run_id,
                message: "terminal runs must have a cache profile row".to_string(),
                repair_hint: Some("import or rewrite the cache profile for this run".to_string()),
                details: json!({"state": state, "cache_profiles": counts.cache_profiles}),
            }));
        }
        if counts.event_stream_events == 0 {
            violations.push(InvariantViolation::new(InvariantViolationInput {
                run_id,
                invariant_id: "terminal_run_has_event_stream",
                severity: "error",
                subject_type: "optimization_runs",
                subject_id: run_id,
                message: "terminal runs must have event stream rows".to_string(),
                repair_hint: Some(
                    "import or rewrite the event stream rows for this run".to_string(),
                ),
                details: json!({"state": state, "event_stream_events": counts.event_stream_events}),
            }));
        }
        if counts.usage_ledger == 0 {
            violations.push(InvariantViolation::new(InvariantViolationInput {
                run_id,
                invariant_id: "terminal_run_has_usage_ledger",
                severity: "error",
                subject_type: "optimization_runs",
                subject_id: run_id,
                message: "terminal runs must have usage ledger rows".to_string(),
                repair_hint: Some(
                    "rebuild usage ledger rows from rollout sensors and proposer events"
                        .to_string(),
                ),
                details: json!({"state": state, "usage_ledger": counts.usage_ledger}),
            }));
        }
        if counts.stopper_states == 0 {
            violations.push(InvariantViolation::new(InvariantViolationInput {
                run_id,
                invariant_id: "terminal_run_has_stopper_state",
                severity: "error",
                subject_type: "optimization_runs",
                subject_id: run_id,
                message: "terminal runs must have stopper/budget state rows".to_string(),
                repair_hint: Some(
                    "rebuild stopper state rows from run budget snapshots".to_string(),
                ),
                details: json!({"state": state, "stopper_states": counts.stopper_states}),
            }));
        }
        if counts.checkpoints == 0 {
            violations.push(InvariantViolation::new(InvariantViolationInput {
                run_id,
                invariant_id: "terminal_run_has_checkpoint",
                severity: "error",
                subject_type: "optimization_runs",
                subject_id: run_id,
                message: "terminal runs must have checkpoint rows".to_string(),
                repair_hint: Some("rebuild checkpoint rows from run snapshots".to_string()),
                details: json!({"state": state, "checkpoints": counts.checkpoints}),
            }));
        }
        if counts.manifests == 0 {
            violations.push(InvariantViolation::new(InvariantViolationInput {
                run_id,
                invariant_id: "terminal_run_has_manifest",
                severity: "error",
                subject_type: "optimization_runs",
                subject_id: run_id,
                message: "terminal runs must have a result manifest row".to_string(),
                repair_hint: Some("import or rewrite the result manifest for this run".to_string()),
                details: json!({"state": state, "manifests": counts.manifests}),
            }));
        }
        if best_candidate_id.as_deref().unwrap_or("").trim().is_empty() && state == "completed" {
            violations.push(InvariantViolation::new(InvariantViolationInput {
                run_id,
                invariant_id: "completed_run_has_best_candidate",
                severity: "error",
                subject_type: "optimization_runs",
                subject_id: run_id,
                message: "completed runs must record a best candidate id".to_string(),
                repair_hint: Some("repair the final selection and manifest import".to_string()),
                details: json!({"state": state}),
            }));
        }
        if counts.active_resource_leases > 0 {
            violations.push(InvariantViolation::new(InvariantViolationInput {
                run_id,
                invariant_id: "terminal_run_releases_resource_leases",
                severity: "error",
                subject_type: "resource_leases",
                subject_id: run_id,
                message: "terminal runs must not have active resource leases".to_string(),
                repair_hint: Some(
                    "release or expire active resource leases for this run".to_string(),
                ),
                details: json!({
                    "state": state,
                    "active_resource_leases": counts.active_resource_leases,
                }),
            }));
        }
        if counts.active_budget_reservations > 0 {
            violations.push(InvariantViolation::new(InvariantViolationInput {
                run_id,
                invariant_id: "terminal_run_releases_budget_reservations",
                severity: "error",
                subject_type: "budget_reservations",
                subject_id: run_id,
                message: "terminal runs must not have active budget reservations".to_string(),
                repair_hint: Some(
                    "commit, release, fail, or cancel active budget reservations for this run"
                        .to_string(),
                ),
                details: json!({
                    "state": state,
                    "active_budget_reservations": counts.active_budget_reservations,
                }),
            }));
        }
    }
    InvariantReport::new(
        run_id,
        checked_at,
        violations,
        json!({
            "state": state,
            "snapshots": {
                "resolved_run_configs": counts.resolved_run_configs,
                "container_contract_snapshots": counts.container_contract_snapshots,
                "prompt_program_snapshots": counts.prompt_program_snapshots,
                "dataset_snapshots": counts.dataset_snapshots,
                "run_limits": counts.run_limits,
            },
            "runtime": {
                "runtime_effects": counts.runtime_effects,
                "runtime_effect_admissions": counts.runtime_effect_admissions,
                "budget_reservations": counts.budget_reservations,
                "budget_commits": counts.budget_commits,
                "budget_releases": counts.budget_releases,
                "active_budget_reservations": counts.active_budget_reservations,
                "rendered_optimizer_states": counts.rendered_optimizer_states,
                "event_stream_events": counts.event_stream_events,
                "usage_ledger": counts.usage_ledger,
                "usage_ledger_expected": counts.usage_ledger_expected,
                "stopper_states": counts.stopper_states,
                "checkpoints": counts.checkpoints,
                "state_transitions": counts.state_transitions,
            },
            "candidates": {
                "candidates": counts.candidates,
                "parented_candidates": counts.parented_candidates,
                "train_frontier_candidates": counts.train_frontier_candidates,
                "candidate_payloads": counts.candidate_payloads,
                "candidate_deltas": counts.candidate_deltas,
                "acceptance_decisions": counts.acceptance_decisions,
                "frontier_cells": counts.frontier_cells,
                "plan_links": counts.plan_links,
            },
            "evaluation": {
                "cache_profiles": counts.cache_profiles,
                "cache_accesses": counts.cache_accesses,
                "materializations": counts.materializations,
                "evaluation_cache": counts.evaluation_cache,
                "evaluation_cache_expected": counts.evaluation_cache_expected,
                "cache_profile_access_total": counts.cache_profile_access_total,
                "sensor_frames": counts.sensor_frames,
                "rollout_jobs": counts.rollout_jobs,
                "rollouts": counts.rollouts,
                "rollout_events": counts.rollout_events,
            },
            "objectives": {
                "objective_sets": counts.objective_sets,
                "objectives": counts.objectives,
                "scores": counts.scores,
                "score_vector_sources": counts.score_vector_sources,
                "score_vectors": counts.score_vectors,
                "pareto_comparisons": counts.pareto_comparisons,
            },
            "evidence": {
                "trace_annotations": counts.trace_annotations,
                "evidence_frames": counts.evidence_frames,
                "verifier_jobs": counts.verifier_jobs,
                "subagent_invocations": counts.subagent_invocations,
                "manifests": counts.manifests,
            },
        }),
    )
}

fn push_exact_count_violation(
    violations: &mut Vec<InvariantViolation>,
    run_id: &str,
    invariant_id: &str,
    source_table: &str,
    source_count: u64,
    derived_table: &str,
    derived_count: u64,
) {
    if source_count != derived_count {
        violations.push(InvariantViolation::count_mismatch(CountMismatchInput {
            run_id,
            invariant_id,
            severity: "error",
            source_table,
            source_count,
            derived_table,
            derived_count,
            comparator: "equal",
        }));
    }
}

fn push_at_least_count_violation(
    violations: &mut Vec<InvariantViolation>,
    run_id: &str,
    invariant_id: &str,
    source_table: &str,
    source_count: u64,
    derived_table: &str,
    derived_count: u64,
) {
    if derived_count < source_count {
        violations.push(InvariantViolation::count_mismatch(CountMismatchInput {
            run_id,
            invariant_id,
            severity: "error",
            source_table,
            source_count,
            derived_table,
            derived_count,
            comparator: "cover",
        }));
    }
}

fn is_terminal_run_state(state: &str) -> bool {
    matches!(state, "completed" | "failed" | "cancelled")
}

fn required_string(value: &Value, field: &str) -> Result<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| OptimizerError::Config(format!("candidate record missing {field}")))
}

fn optional_string(value: &Value, field: &str) -> Option<String> {
    value.get(field).and_then(Value::as_str).map(str::to_string)
}

fn parse_json_or_null(raw: Option<&str>) -> Value {
    raw.and_then(|text| serde_json::from_str(text).ok())
        .unwrap_or(Value::Null)
}

fn nonnegative_u64(value: i64) -> u64 {
    value.max(0) as u64
}

fn now_rfc3339() -> String {
    OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}
