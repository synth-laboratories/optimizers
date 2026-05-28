use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::cache::{normalize_for_cache_profile, stable_json, stable_json_hash, stable_value_hash};

pub const EVALUATION_CACHE_SCHEMA_VERSION: &str = "evaluation_cache.v1";
pub const EVALUATION_CACHE_PROFILE: &str = "rollout_request";
pub const EVALUATION_CACHE_KEY_FIELDS_SCHEMA_VERSION: &str = "evaluation_cache_key_fields.v1";
pub const MATERIALIZATION_SCHEMA_VERSION: &str = "materialization.v1";
pub const EVALUATION_CACHE_KEY_FIELD_NAMES: [&str; 10] = [
    "candidate_hash",
    "example_hash",
    "request_hash",
    "example_id",
    "evaluator_id",
    "algorithm_id",
    "materializer_id",
    "lever_version",
    "sensor_version",
    "objective_set_hash",
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RolloutMaterializationIdentity {
    pub evaluator_id: String,
    pub algorithm_id: String,
    pub materializer_id: String,
    pub lever_version: String,
    pub sensor_version: String,
    pub objective_set_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationCacheIdentity {
    pub candidate_hash: String,
    pub example_hash: String,
    pub request_hash: String,
    pub example_id: String,
    pub evaluator_id: String,
    pub algorithm_id: String,
    pub materializer_id: String,
    pub lever_version: String,
    pub sensor_version: String,
    pub objective_set_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationCacheRecord {
    pub schema_version: String,
    pub cache_key: String,
    pub cache_profile: String,
    pub cache_key_fields: Value,
    pub candidate_hash: String,
    pub example_hash: String,
    pub request_hash: String,
    pub example_id: String,
    pub evaluator_id: String,
    pub algorithm_id: String,
    pub materializer_id: String,
    pub lever_version: String,
    pub sensor_version: String,
    pub objective_set_hash: String,
    #[serde(default)]
    pub source_rollout_id: Option<String>,
    pub reward: f64,
    #[serde(default)]
    pub objective_scores: Value,
    #[serde(default)]
    pub actionable_side_info: Value,
    #[serde(default)]
    pub usage: Value,
    #[serde(default)]
    pub trace_ref: Option<String>,
    pub status: String,
    #[serde(default)]
    pub cache_hit: bool,
    #[serde(default)]
    pub platform_cache_key: Option<String>,
    #[serde(default)]
    pub rollout_payload: Value,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaterializationRecord {
    pub schema_version: String,
    pub materialization_id: String,
    pub candidate_id: String,
    pub example_id: String,
    pub seed: i64,
    pub split: String,
    pub evaluation_stage: String,
    pub task_id: String,
    pub evaluator_id: String,
    pub algorithm_id: String,
    pub materializer_id: String,
    pub lever_version: String,
    pub sensor_version: String,
    pub objective_set_hash: String,
    pub candidate_hash: String,
    pub example_hash: String,
    pub request_hash: String,
    pub cache_key: String,
    #[serde(default)]
    pub platform_cache_key: Option<String>,
    pub status: String,
    pub request: Value,
    pub candidate_payload: Value,
    pub dataset_row: Value,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

pub struct EvaluationCacheRecordInput<'a> {
    pub candidate_payload: &'a Value,
    pub example: &'a Value,
    pub request: &'a Value,
    pub example_id: &'a str,
    pub materialization: RolloutMaterializationIdentity,
    pub source_rollout_id: Option<String>,
    pub reward: f64,
    pub objective_scores: Value,
    pub actionable_side_info: Value,
    pub usage: Value,
    pub trace_ref: Option<String>,
    pub status: &'a str,
    pub cache_hit: bool,
    pub platform_cache_key: Option<String>,
    pub rollout_payload: &'a Value,
    pub metadata: Map<String, Value>,
}

pub struct MaterializationRecordInput<'a> {
    pub candidate_id: &'a str,
    pub candidate_payload: &'a Value,
    pub example: &'a Value,
    pub request: &'a Value,
    pub example_id: &'a str,
    pub seed: i64,
    pub split: &'a str,
    pub evaluation_stage: &'a str,
    pub task_id: &'a str,
    pub materialization: RolloutMaterializationIdentity,
    pub status: &'a str,
    pub platform_cache_key: Option<String>,
    pub metadata: Map<String, Value>,
}

impl RolloutMaterializationIdentity {
    pub fn prompt_overlay(
        algorithm_id: &str,
        program_id: &str,
        lever_version: &str,
        objective_set_hash: &str,
    ) -> Self {
        Self {
            evaluator_id: "container.reward_info".to_string(),
            algorithm_id: algorithm_id.to_string(),
            materializer_id: format!("prompt_overlay:{program_id}"),
            lever_version: lever_version.to_string(),
            sensor_version: "sensor_frame.v1".to_string(),
            objective_set_hash: objective_set_hash.to_string(),
        }
    }

    pub fn lever_bundle(
        algorithm_id: &str,
        program_id: &str,
        lever_version: &str,
        objective_set_hash: &str,
    ) -> Self {
        Self {
            evaluator_id: "container.reward_info".to_string(),
            algorithm_id: algorithm_id.to_string(),
            materializer_id: format!("lever_bundle:{program_id}"),
            lever_version: lever_version.to_string(),
            sensor_version: "sensor_frame.v1".to_string(),
            objective_set_hash: objective_set_hash.to_string(),
        }
    }
}

impl EvaluationCacheIdentity {
    pub fn from_parts(
        candidate_payload: &Value,
        example: &Value,
        request: &Value,
        example_id: &str,
        materialization: &RolloutMaterializationIdentity,
    ) -> Self {
        Self {
            candidate_hash: stable_value_hash(candidate_payload),
            example_hash: stable_value_hash(example),
            request_hash: stable_json_hash(&normalize_for_cache_profile(
                request,
                EVALUATION_CACHE_PROFILE,
            )),
            example_id: example_id.to_string(),
            evaluator_id: materialization.evaluator_id.clone(),
            algorithm_id: materialization.algorithm_id.clone(),
            materializer_id: materialization.materializer_id.clone(),
            lever_version: materialization.lever_version.clone(),
            sensor_version: materialization.sensor_version.clone(),
            objective_set_hash: materialization.objective_set_hash.clone(),
        }
    }

    pub fn cache_key(&self) -> String {
        stable_json_hash(&serde_json::to_value(self).unwrap_or(Value::Null))
    }
}

impl EvaluationCacheRecord {
    pub fn from_input(input: EvaluationCacheRecordInput<'_>) -> Self {
        let identity = EvaluationCacheIdentity::from_parts(
            input.candidate_payload,
            input.example,
            input.request,
            input.example_id,
            &input.materialization,
        );
        Self {
            schema_version: EVALUATION_CACHE_SCHEMA_VERSION.to_string(),
            cache_key: identity.cache_key(),
            cache_profile: EVALUATION_CACHE_PROFILE.to_string(),
            cache_key_fields: evaluation_cache_key_fields(),
            candidate_hash: identity.candidate_hash,
            example_hash: identity.example_hash,
            request_hash: identity.request_hash,
            example_id: identity.example_id,
            evaluator_id: identity.evaluator_id,
            algorithm_id: identity.algorithm_id,
            materializer_id: identity.materializer_id,
            lever_version: identity.lever_version,
            sensor_version: identity.sensor_version,
            objective_set_hash: identity.objective_set_hash,
            source_rollout_id: input.source_rollout_id,
            reward: input.reward,
            objective_scores: input.objective_scores,
            actionable_side_info: input.actionable_side_info,
            usage: input.usage,
            trace_ref: input.trace_ref,
            status: input.status.to_string(),
            cache_hit: input.cache_hit,
            platform_cache_key: input.platform_cache_key,
            rollout_payload: input.rollout_payload.clone(),
            metadata: input.metadata,
        }
    }
}

impl MaterializationRecord {
    pub fn from_input(input: MaterializationRecordInput<'_>) -> Self {
        let identity = EvaluationCacheIdentity::from_parts(
            input.candidate_payload,
            input.example,
            input.request,
            input.example_id,
            &input.materialization,
        );
        let materialization_identity = json!({
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "candidate_id": input.candidate_id,
            "example_id": input.example_id,
            "seed": input.seed,
            "split": input.split,
            "evaluation_stage": input.evaluation_stage,
            "task_id": input.task_id,
            "evaluator_id": identity.evaluator_id.clone(),
            "algorithm_id": identity.algorithm_id.clone(),
            "materializer_id": identity.materializer_id.clone(),
            "lever_version": identity.lever_version.clone(),
            "sensor_version": identity.sensor_version.clone(),
            "objective_set_hash": identity.objective_set_hash.clone(),
            "candidate_hash": identity.candidate_hash.clone(),
            "example_hash": identity.example_hash.clone(),
            "request_hash": identity.request_hash.clone(),
        });
        let cache_key = identity.cache_key();
        Self {
            schema_version: MATERIALIZATION_SCHEMA_VERSION.to_string(),
            materialization_id: prefixed_hash_id("mat", &materialization_identity),
            candidate_id: input.candidate_id.to_string(),
            example_id: identity.example_id.clone(),
            seed: input.seed,
            split: input.split.to_string(),
            evaluation_stage: input.evaluation_stage.to_string(),
            task_id: input.task_id.to_string(),
            evaluator_id: identity.evaluator_id,
            algorithm_id: identity.algorithm_id,
            materializer_id: identity.materializer_id,
            lever_version: identity.lever_version,
            sensor_version: identity.sensor_version,
            objective_set_hash: identity.objective_set_hash,
            candidate_hash: identity.candidate_hash,
            example_hash: identity.example_hash,
            request_hash: identity.request_hash,
            cache_key,
            platform_cache_key: input.platform_cache_key,
            status: input.status.to_string(),
            request: input.request.clone(),
            candidate_payload: input.candidate_payload.clone(),
            dataset_row: input.example.clone(),
            metadata: input.metadata,
        }
    }
}

pub fn evaluation_cache_key_fields() -> Value {
    json!({
        "schema_version": EVALUATION_CACHE_KEY_FIELDS_SCHEMA_VERSION,
        "fields": EVALUATION_CACHE_KEY_FIELD_NAMES,
    })
}

pub fn objective_set_hash(objective_scores: &Value) -> String {
    let value = json!({
        "schema_version": "objective_set_identity.v1",
        "objectives": objective_scores,
    });
    stable_value_hash(&value)
}

pub fn record_json(record: &EvaluationCacheRecord) -> String {
    stable_json(&serde_json::to_value(record).unwrap_or(Value::Null))
}

pub fn materialization_record_json(record: &MaterializationRecord) -> String {
    stable_json(&serde_json::to_value(record).unwrap_or(Value::Null))
}

fn prefixed_hash_id(prefix: &str, value: &Value) -> String {
    let hash = stable_value_hash(value);
    format!("{prefix}_{}", &hash[..16])
}
