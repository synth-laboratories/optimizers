use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use time::OffsetDateTime;

use crate::cache::{stable_json, stable_value_hash};

pub const RESOLVED_RUN_CONFIG_SCHEMA_VERSION: &str = "resolved_run_config.v1";
pub const CONTAINER_CONTRACT_SNAPSHOT_SCHEMA_VERSION: &str = "container_contract_snapshot.v1";
pub const PROMPT_PROGRAM_SNAPSHOT_SCHEMA_VERSION: &str = "prompt_program_snapshot.v1";
pub const TASKSET_SNAPSHOT_SCHEMA_VERSION: &str = "taskset_snapshot.v1";
pub const RENDERED_OPTIMIZER_STATE_SCHEMA_VERSION: &str = "rendered_optimizer_state.v1";
pub const RUNTIME_EFFECT_SCHEMA_VERSION: &str = "runtime_effect.v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResolvedRunConfigRecord {
    pub schema_version: String,
    pub resolved_config_id: String,
    pub run_id: String,
    pub algorithm_id: String,
    pub config_hash: String,
    pub cache_mode: String,
    pub cache_namespace: String,
    pub output_dir: String,
    pub config: Value,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    pub recorded_at: String,
}

pub struct ResolvedRunConfigInput<'a> {
    pub run_id: &'a str,
    pub algorithm_id: &'a str,
    pub cache_mode: &'a str,
    pub cache_namespace: &'a str,
    pub output_dir: &'a str,
    pub config: &'a Value,
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContainerContractSnapshotRecord {
    pub schema_version: String,
    pub contract_snapshot_id: String,
    pub run_id: String,
    pub container_url: String,
    pub contract_kind: String,
    pub contract_version: String,
    pub validation_status: String,
    pub capability_hash: String,
    pub metadata_response: Value,
    #[serde(default)]
    pub health_response: Option<Value>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    pub recorded_at: String,
}

pub struct ContainerContractSnapshotInput<'a> {
    pub run_id: &'a str,
    pub container_url: &'a str,
    pub contract_kind: &'a str,
    pub contract_version: &'a str,
    pub validation_status: &'a str,
    pub metadata_response: &'a Value,
    pub health_response: Option<Value>,
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromptProgramSnapshotRecord {
    pub schema_version: String,
    pub program_snapshot_id: String,
    pub run_id: String,
    pub program_id: String,
    pub program_hash: String,
    #[serde(default)]
    pub target_modules: Vec<String>,
    #[serde(default)]
    pub mutable_field_ids: Vec<String>,
    pub validation_status: String,
    pub program: Value,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    pub recorded_at: String,
}

pub struct PromptProgramSnapshotInput<'a> {
    pub run_id: &'a str,
    pub program_id: &'a str,
    pub target_modules: &'a [String],
    pub mutable_field_ids: Vec<String>,
    pub validation_status: &'a str,
    pub program: &'a Value,
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TasksetSnapshotRecord {
    pub schema_version: String,
    pub taskset_snapshot_id: String,
    pub run_id: String,
    pub taskset_id: String,
    pub split: String,
    pub task_count: u64,
    pub task_id_count: u64,
    #[serde(default)]
    pub task_ids: Vec<String>,
    #[serde(default)]
    pub filters: Value,
    pub tasks_hash: String,
    pub tasks: Value,
    #[serde(default)]
    pub taskset_metadata: Value,
    #[serde(default)]
    pub tasks_metadata: Value,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    pub recorded_at: String,
}

pub struct TasksetSnapshotInput<'a> {
    pub run_id: &'a str,
    pub taskset_id: &'a str,
    pub split: &'a str,
    pub task_ids: &'a [String],
    pub filters: &'a Value,
    pub tasks: &'a [Value],
    pub taskset_metadata: Value,
    pub tasks_metadata: Value,
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderedOptimizerStateRecord {
    pub schema_version: String,
    pub rendered_state_id: String,
    pub run_id: String,
    pub sequence_number: u64,
    pub run_phase: String,
    #[serde(default)]
    pub generation_phase: Option<String>,
    #[serde(default)]
    pub candidate_phase: Option<String>,
    pub block_status: String,
    #[serde(default)]
    pub terminal_status: Option<String>,
    #[serde(default)]
    pub best_candidate_id: Option<String>,
    pub frontier_size: u64,
    pub active_effect_count: u64,
    pub active_job_count: u64,
    #[serde(default)]
    pub queue_counts: Value,
    #[serde(default)]
    pub budget_status: Value,
    #[serde(default)]
    pub evidence_status: Value,
    #[serde(default)]
    pub details: Value,
    pub rendered_at: String,
}

pub struct RenderedOptimizerStateInput<'a> {
    pub run_id: &'a str,
    pub sequence_number: u64,
    pub run_phase: &'a str,
    pub generation_phase: Option<String>,
    pub candidate_phase: Option<String>,
    pub block_status: &'a str,
    pub terminal_status: Option<String>,
    pub best_candidate_id: Option<String>,
    pub frontier_size: u64,
    pub active_effect_count: u64,
    pub active_job_count: u64,
    pub queue_counts: Value,
    pub budget_status: Value,
    pub evidence_status: Value,
    pub details: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeEffectRecord {
    pub schema_version: String,
    pub runtime_effect_id: String,
    pub run_id: String,
    pub effect_kind: String,
    pub lane: String,
    pub status: String,
    pub subject_type: String,
    pub subject_id: String,
    pub idempotency_key: String,
    #[serde(default)]
    pub cache_key: Option<String>,
    #[serde(default)]
    pub job_id: Option<String>,
    #[serde(default)]
    pub budget_reservation_id: Option<String>,
    pub attempt: u32,
    #[serde(default)]
    pub failure_class: Option<String>,
    #[serde(default)]
    pub payload: Value,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    pub planned_at: String,
    pub updated_at: String,
    #[serde(default)]
    pub terminal_at: Option<String>,
}

pub struct RuntimeEffectInput<'a> {
    pub run_id: &'a str,
    pub effect_kind: &'a str,
    pub lane: &'a str,
    pub status: &'a str,
    pub subject_type: &'a str,
    pub subject_id: &'a str,
    pub idempotency_key: &'a str,
    pub cache_key: Option<String>,
    pub job_id: Option<String>,
    pub budget_reservation_id: Option<String>,
    pub attempt: u32,
    pub failure_class: Option<String>,
    pub payload: Value,
    pub metadata: Map<String, Value>,
}

impl ResolvedRunConfigRecord {
    pub fn from_input(input: ResolvedRunConfigInput<'_>) -> Self {
        let config_hash = stable_value_hash(input.config);
        let identity = json!({
            "run_id": input.run_id,
            "algorithm_id": input.algorithm_id,
            "config_hash": config_hash,
        });
        Self {
            schema_version: RESOLVED_RUN_CONFIG_SCHEMA_VERSION.to_string(),
            resolved_config_id: prefixed_hash_id("resolved_config", &identity),
            run_id: input.run_id.to_string(),
            algorithm_id: input.algorithm_id.to_string(),
            config_hash,
            cache_mode: input.cache_mode.to_string(),
            cache_namespace: input.cache_namespace.to_string(),
            output_dir: input.output_dir.to_string(),
            config: input.config.clone(),
            metadata: input.metadata,
            recorded_at: now_rfc3339(),
        }
    }
}

impl ContainerContractSnapshotRecord {
    pub fn from_input(input: ContainerContractSnapshotInput<'_>) -> Self {
        let capability_hash = stable_value_hash(input.metadata_response);
        let identity = json!({
            "run_id": input.run_id,
            "container_url": input.container_url,
            "contract_kind": input.contract_kind,
            "contract_version": input.contract_version,
            "capability_hash": capability_hash,
        });
        Self {
            schema_version: CONTAINER_CONTRACT_SNAPSHOT_SCHEMA_VERSION.to_string(),
            contract_snapshot_id: prefixed_hash_id("contract", &identity),
            run_id: input.run_id.to_string(),
            container_url: input.container_url.to_string(),
            contract_kind: input.contract_kind.to_string(),
            contract_version: input.contract_version.to_string(),
            validation_status: input.validation_status.to_string(),
            capability_hash,
            metadata_response: input.metadata_response.clone(),
            health_response: input.health_response,
            metadata: input.metadata,
            recorded_at: now_rfc3339(),
        }
    }
}

impl PromptProgramSnapshotRecord {
    pub fn from_input(input: PromptProgramSnapshotInput<'_>) -> Self {
        let program_hash = stable_value_hash(input.program);
        let identity = json!({
            "run_id": input.run_id,
            "program_id": input.program_id,
            "program_hash": program_hash,
            "target_modules": input.target_modules,
        });
        Self {
            schema_version: PROMPT_PROGRAM_SNAPSHOT_SCHEMA_VERSION.to_string(),
            program_snapshot_id: prefixed_hash_id("program", &identity),
            run_id: input.run_id.to_string(),
            program_id: input.program_id.to_string(),
            program_hash,
            target_modules: input.target_modules.to_vec(),
            mutable_field_ids: input.mutable_field_ids,
            validation_status: input.validation_status.to_string(),
            program: input.program.clone(),
            metadata: input.metadata,
            recorded_at: now_rfc3339(),
        }
    }
}

impl TasksetSnapshotRecord {
    pub fn from_input(input: TasksetSnapshotInput<'_>) -> Self {
        let tasks = Value::Array(input.tasks.to_vec());
        let tasks_hash = stable_value_hash(&tasks);
        let identity = json!({
            "run_id": input.run_id,
            "taskset_id": input.taskset_id,
            "split": input.split,
            "task_ids": input.task_ids,
            "filters": input.filters,
            "tasks_hash": tasks_hash,
        });
        Self {
            schema_version: TASKSET_SNAPSHOT_SCHEMA_VERSION.to_string(),
            taskset_snapshot_id: prefixed_hash_id("taskset", &identity),
            run_id: input.run_id.to_string(),
            taskset_id: input.taskset_id.to_string(),
            split: input.split.to_string(),
            task_count: input.tasks.len() as u64,
            task_id_count: input.task_ids.len() as u64,
            task_ids: input.task_ids.to_vec(),
            filters: input.filters.clone(),
            tasks_hash,
            tasks,
            taskset_metadata: input.taskset_metadata,
            tasks_metadata: input.tasks_metadata,
            metadata: input.metadata,
            recorded_at: now_rfc3339(),
        }
    }
}

impl RenderedOptimizerStateRecord {
    pub fn from_input(input: RenderedOptimizerStateInput<'_>) -> Self {
        let identity = json!({
            "schema_version": RENDERED_OPTIMIZER_STATE_SCHEMA_VERSION,
            "run_id": input.run_id,
            "sequence_number": input.sequence_number,
        });
        let hash = stable_value_hash(&identity);
        Self {
            schema_version: RENDERED_OPTIMIZER_STATE_SCHEMA_VERSION.to_string(),
            rendered_state_id: format!("rendered_state_{}", &hash[..16]),
            run_id: input.run_id.to_string(),
            sequence_number: input.sequence_number,
            run_phase: input.run_phase.to_string(),
            generation_phase: input.generation_phase,
            candidate_phase: input.candidate_phase,
            block_status: input.block_status.to_string(),
            terminal_status: input.terminal_status,
            best_candidate_id: input.best_candidate_id,
            frontier_size: input.frontier_size,
            active_effect_count: input.active_effect_count,
            active_job_count: input.active_job_count,
            queue_counts: input.queue_counts,
            budget_status: input.budget_status,
            evidence_status: input.evidence_status,
            details: input.details,
            rendered_at: now_rfc3339(),
        }
    }
}

impl RuntimeEffectRecord {
    pub fn from_input(input: RuntimeEffectInput<'_>) -> Self {
        let identity = json!({
            "run_id": input.run_id,
            "effect_kind": input.effect_kind,
            "subject_type": input.subject_type,
            "subject_id": input.subject_id,
            "idempotency_key": input.idempotency_key,
            "attempt": input.attempt,
        });
        let now = now_rfc3339();
        let terminal_at = if matches!(
            input.status,
            "completed" | "failed" | "cancelled" | "canceled" | "expired" | "rejected"
        ) {
            Some(now.clone())
        } else {
            None
        };
        Self {
            schema_version: RUNTIME_EFFECT_SCHEMA_VERSION.to_string(),
            runtime_effect_id: prefixed_hash_id("effect", &identity),
            run_id: input.run_id.to_string(),
            effect_kind: input.effect_kind.to_string(),
            lane: input.lane.to_string(),
            status: input.status.to_string(),
            subject_type: input.subject_type.to_string(),
            subject_id: input.subject_id.to_string(),
            idempotency_key: input.idempotency_key.to_string(),
            cache_key: input.cache_key,
            job_id: input.job_id,
            budget_reservation_id: input.budget_reservation_id,
            attempt: input.attempt,
            failure_class: input.failure_class,
            payload: input.payload,
            metadata: input.metadata,
            planned_at: now.clone(),
            updated_at: now,
            terminal_at,
        }
    }
}

pub fn runtime_record_json<T: Serialize>(record: &T) -> String {
    stable_json(&serde_json::to_value(record).unwrap_or(Value::Null))
}

fn prefixed_hash_id(prefix: &str, value: &Value) -> String {
    let hash = stable_value_hash(value);
    format!("{prefix}_{}", &hash[..16])
}

fn now_rfc3339() -> String {
    OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}
