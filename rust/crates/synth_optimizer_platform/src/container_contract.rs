use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::error::{OptimizerError, Result};
use crate::prompt_program::PromptProgram;
use crate::GEPA_OPTIMIZER_CONTRACT_VERSION;

pub type JsonMap = Map<String, Value>;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HealthResponse {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub contract_version: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ContainerMetadataResponse {
    #[serde(default)]
    pub runtime: JsonMap,
    #[serde(default)]
    pub capabilities: JsonMap,
    #[serde(default)]
    pub metadata: ContainerMetadata,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl ContainerMetadataResponse {
    pub fn gepa_contract(&self) -> Result<&GepaOptimizerContract> {
        self.metadata
            .optimizer_contracts
            .gepa
            .as_ref()
            .ok_or_else(|| {
                OptimizerError::Container(
                    "container metadata must advertise metadata.optimizer_contracts.gepa"
                        .to_string(),
                )
            })
    }

    pub fn validate_gepa_contract(&self) -> Result<()> {
        let contract = self.gepa_contract()?;
        if contract.version != GEPA_OPTIMIZER_CONTRACT_VERSION {
            return Err(OptimizerError::Container(format!(
                "container does not advertise metadata.optimizer_contracts.gepa.version={}",
                GEPA_OPTIMIZER_CONTRACT_VERSION
            )));
        }
        contract.validate_routes()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ContainerMetadata {
    #[serde(default)]
    pub optimizer_contracts: OptimizerContracts,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OptimizerContracts {
    #[serde(default)]
    pub gepa: Option<GepaOptimizerContract>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GepaOptimizerContract {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub program_route: String,
    #[serde(default)]
    pub dataset_route: String,
    #[serde(default)]
    pub dataset_rows_route: String,
    #[serde(default)]
    pub rollout_route: String,
    #[serde(default)]
    pub trace_route: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl GepaOptimizerContract {
    fn validate_routes(&self) -> Result<()> {
        for (name, route) in [
            ("program_route", self.program_route.as_str()),
            ("dataset_route", self.dataset_route.as_str()),
            ("dataset_rows_route", self.dataset_rows_route.as_str()),
            ("rollout_route", self.rollout_route.as_str()),
        ] {
            if !route.starts_with('/') {
                return Err(OptimizerError::Container(format!(
                    "metadata.optimizer_contracts.gepa.{name} must be an absolute route, got {route:?}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DatasetResponse {
    #[serde(default)]
    pub dataset_id: Option<String>,
    #[serde(default)]
    pub splits: JsonMap,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub metadata: JsonMap,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetRowsRequest {
    pub split: String,
    #[serde(default)]
    pub seeds: Vec<i64>,
    #[serde(default)]
    pub filters: Value,
}

impl DatasetRowsRequest {
    pub fn new(split: impl Into<String>, seeds: &[i64], filters: Value) -> Self {
        Self {
            split: split.into(),
            seeds: seeds.to_vec(),
            filters,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DatasetRowsResponse {
    #[serde(default)]
    pub rows: Vec<Value>,
    #[serde(default)]
    pub metadata: JsonMap,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl DatasetRowsResponse {
    pub fn validate_for_request(&self, request: &DatasetRowsRequest) -> Result<()> {
        if self.rows.len() != request.seeds.len() {
            return Err(OptimizerError::Container(format!(
                "/dataset/rows returned {} rows for {} requested seeds on split {:?}",
                self.rows.len(),
                request.seeds.len(),
                request.split
            )));
        }
        let mut identities = BTreeSet::new();
        for (index, row) in self.rows.iter().enumerate() {
            if !row.is_object() {
                return Err(OptimizerError::Container(format!(
                    "/dataset/rows row {index} must be an object"
                )));
            }
            let identity = dataset_row_identity(row).map_err(|_| {
                OptimizerError::Container(format!(
                    "/dataset/rows row {index} must include a stable row identity: example_id, id, task_instance_id, seed, or task_id+index"
                ))
            })?;
            if !identities.insert(identity.clone()) {
                return Err(OptimizerError::Container(format!(
                    "/dataset/rows row {index} has duplicate stable row identity {identity:?}"
                )));
            }
        }
        Ok(())
    }
}

pub fn dataset_row_identity(row: &Value) -> Result<String> {
    let object = row
        .as_object()
        .ok_or_else(|| OptimizerError::Container("dataset row must be an object".to_string()))?;
    for key in ["example_id", "id", "task_instance_id"] {
        if let Some(identity) = identity_part(object.get(key)) {
            return Ok(identity);
        }
    }
    if let Some(seed) = object.get("seed").and_then(Value::as_i64) {
        let split = object
            .get("split")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("seed");
        return Ok(format!("{split}:{seed}"));
    }
    if let (Some(task_id), Some(index)) = (
        identity_part(object.get("task_id")),
        object.get("index").and_then(Value::as_i64),
    ) {
        return Ok(format!("{task_id}:{index}"));
    }
    Err(OptimizerError::Container(
        "dataset row must include a stable row identity: example_id, id, task_instance_id, seed, or task_id+index".to_string(),
    ))
}

fn identity_part(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RolloutActorSpec {
    pub actor_id: String,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub config: JsonMap,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RolloutRequest {
    #[serde(default)]
    pub rollout_id: Option<String>,
    #[serde(default)]
    pub trace_correlation_id: Option<String>,
    #[serde(default)]
    pub trial_id: Option<String>,
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub submission_mode: Option<String>,
    #[serde(default)]
    pub env: Value,
    #[serde(default)]
    pub policy: Value,
    #[serde(default)]
    pub candidate: JsonMap,
    #[serde(default)]
    pub candidate_overlay: JsonMap,
    #[serde(default)]
    pub dataset_row: JsonMap,
    #[serde(default)]
    pub dataset: Value,
    #[serde(default)]
    pub long_horizon: JsonMap,
    #[serde(default)]
    pub terminator: JsonMap,
    #[serde(default)]
    pub task_payload: JsonMap,
    #[serde(default)]
    pub task_id: Option<String>,
    #[serde(default)]
    pub task_instance_id: Option<String>,
    #[serde(default)]
    pub task_metadata: JsonMap,
    #[serde(default)]
    pub checkpoint: Option<Value>,
    #[serde(default)]
    pub checkpoint_id: Option<String>,
    #[serde(default)]
    pub checkpoint_data_base64: Option<String>,
    #[serde(default)]
    pub target_rollout_id: Option<String>,
    #[serde(default)]
    pub actors: Vec<RolloutActorSpec>,
    #[serde(default)]
    pub actor_ids: Vec<String>,
    #[serde(default)]
    pub actor_overrides: JsonMap,
    #[serde(default)]
    pub metadata: JsonMap,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RewardInfo {
    #[serde(default)]
    pub outcome_reward: Option<f64>,
    #[serde(default)]
    pub event_rewards: Vec<Value>,
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(default)]
    pub metrics: JsonMap,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl RewardInfo {
    pub fn reward_value(&self) -> Option<f64> {
        self.outcome_reward.or(self.score)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ArtifactItem {
    #[serde(default)]
    pub artifact_id: Option<String>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub uri: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub media_type: Option<String>,
    #[serde(default)]
    pub digest: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub metadata: JsonMap,
    #[serde(flatten)]
    pub extra: JsonMap,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RolloutResponse {
    #[serde(default)]
    pub trace_correlation_id: Option<String>,
    #[serde(default)]
    pub rollout_id: Option<String>,
    #[serde(default)]
    pub trial_id: Option<String>,
    #[serde(default = "default_rollout_status")]
    pub status: String,
    #[serde(default)]
    pub success_status: Option<String>,
    #[serde(default)]
    pub status_detail: Option<String>,
    #[serde(default)]
    pub summary: JsonMap,
    #[serde(default)]
    pub reward_info: Option<RewardInfo>,
    #[serde(default)]
    pub trace: Option<Value>,
    #[serde(default)]
    pub turns: Vec<Value>,
    #[serde(default)]
    pub artifacts: Vec<ArtifactItem>,
    #[serde(default)]
    pub events: Vec<Value>,
    #[serde(default)]
    pub usage: JsonMap,
    #[serde(default)]
    pub checkpoint: Option<Value>,
    #[serde(default)]
    pub checkpoint_id: Option<String>,
    #[serde(default)]
    pub task_id: Option<String>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub updated_at: Option<String>,
    #[serde(default)]
    pub parent_rollout_id: Option<String>,
    #[serde(default)]
    pub parent_checkpoint_id: Option<String>,
    #[serde(default)]
    pub metadata: JsonMap,
    #[serde(default)]
    pub source: JsonMap,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl RolloutResponse {
    pub fn from_value(value: Value) -> Result<Self> {
        Ok(serde_json::from_value(value)?)
    }

    pub fn validate_for_gepa(&self) -> Result<()> {
        let rollout_id = self.rollout_id.as_deref().unwrap_or_default();
        if rollout_id.trim().is_empty() {
            return Err(OptimizerError::Container(
                "/rollout response must include rollout_id".to_string(),
            ));
        }
        let _ = self.outcome_reward()?;
        Ok(())
    }

    pub fn outcome_reward(&self) -> Result<f64> {
        if let Some(reward) = self.reward_info.as_ref().and_then(RewardInfo::reward_value) {
            return Ok(reward);
        }
        if let Some(reward) = self.summary.get("outcome_reward").and_then(Value::as_f64) {
            return Ok(reward);
        }
        if let Some(reward) = self.extra.get("reward").and_then(Value::as_f64) {
            return Ok(reward);
        }
        Err(OptimizerError::Container(
            "rollout response must include numeric reward_info.outcome_reward, reward_info.score, summary.outcome_reward, or reward".to_string(),
        ))
    }

    pub fn has_v4_trace(&self) -> bool {
        self.trace
            .as_ref()
            .and_then(|trace| trace.get("schema_version"))
            .and_then(Value::as_str)
            == Some(TRACE_SCHEMA_VERSION_NAME)
    }
}

fn default_rollout_status() -> String {
    "completed".to_string()
}

pub const TRACE_SCHEMA_VERSION: u64 = 4;
pub const TRACE_SCHEMA_VERSION_NAME: &str = "synth_rollout_trace_v4";

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RolloutTraceV4 {
    pub rollout_id: String,
    #[serde(default)]
    pub spans: Vec<RolloutTraceSpanV4>,
    #[serde(default)]
    pub trace_correlation_id: Option<String>,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub summary: JsonMap,
    #[serde(default)]
    pub events: Vec<Value>,
    #[serde(default)]
    pub metadata: JsonMap,
    #[serde(default = "trace_schema_version_name")]
    pub schema_version: String,
    #[serde(default = "trace_schema_version")]
    pub trace_schema_version: u64,
    #[serde(default)]
    pub event_history: Vec<Value>,
    #[serde(default)]
    pub span_count: Option<usize>,
}

impl RolloutTraceV4 {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != TRACE_SCHEMA_VERSION_NAME {
            return Err(OptimizerError::Container(format!(
                "rollout trace schema_version must be {TRACE_SCHEMA_VERSION_NAME:?}, got {:?}",
                self.schema_version
            )));
        }
        if self.trace_schema_version != TRACE_SCHEMA_VERSION {
            return Err(OptimizerError::Container(format!(
                "rollout trace trace_schema_version must be {TRACE_SCHEMA_VERSION}, got {}",
                self.trace_schema_version
            )));
        }
        if self.rollout_id.trim().is_empty() {
            return Err(OptimizerError::Container(
                "rollout trace rollout_id is required".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RolloutTraceSpanV4 {
    pub span_id: String,
    pub call_index: u64,
    pub request: CanonicalRequest,
    pub response: CanonicalResponse,
    #[serde(default)]
    pub parent_span_id: Option<String>,
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub api_format: Option<String>,
    #[serde(default)]
    pub raw_request: Option<Value>,
    #[serde(default)]
    pub raw_response: Option<Value>,
    #[serde(default)]
    pub metrics: JsonMap,
    #[serde(default)]
    pub metadata: JsonMap,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CanonicalRequest {
    #[serde(default)]
    pub messages: Vec<CanonicalMessage>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub response_format: Option<Value>,
    #[serde(default)]
    pub provider_hint: Option<String>,
    #[serde(default = "trace_schema_version")]
    pub schema_version: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CanonicalResponse {
    #[serde(default)]
    pub choices: Vec<CanonicalChoice>,
    #[serde(default)]
    pub usage: CanonicalUsage,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub response_id: Option<String>,
    #[serde(default)]
    pub created_at: Option<f64>,
    #[serde(default)]
    pub provider_hint: Option<String>,
    #[serde(default = "trace_schema_version")]
    pub schema_version: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CanonicalChoice {
    pub index: u64,
    pub message: CanonicalMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CanonicalMessage {
    pub role: String,
    #[serde(default)]
    pub parts: Vec<Value>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CanonicalUsage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub reasoning_tokens: u64,
    #[serde(default)]
    pub cached_tokens: u64,
}

fn trace_schema_version_name() -> String {
    TRACE_SCHEMA_VERSION_NAME.to_string()
}

fn trace_schema_version() -> u64 {
    TRACE_SCHEMA_VERSION
}

pub fn decode_container_metadata(value: Value) -> Result<ContainerMetadataResponse> {
    Ok(serde_json::from_value(value)?)
}

pub fn decode_prompt_program(value: Value) -> Result<PromptProgram> {
    Ok(serde_json::from_value(value)?)
}

pub fn decode_dataset_rows(
    value: Value,
    request: &DatasetRowsRequest,
) -> Result<DatasetRowsResponse> {
    let rows: DatasetRowsResponse = serde_json::from_value(value)?;
    rows.validate_for_request(request)?;
    Ok(rows)
}

pub fn decode_rollout_response(value: Value) -> Result<RolloutResponse> {
    let response = RolloutResponse::from_value(value)?;
    response.validate_for_gepa()?;
    Ok(response)
}

pub fn rollout_response_value(response: &RolloutResponse) -> Value {
    serde_json::to_value(response).unwrap_or_else(|_| json!({}))
}
