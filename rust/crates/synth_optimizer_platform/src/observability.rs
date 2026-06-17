use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub const OPTIMIZER_EVENT_SCHEMA_VERSION: &str = "optimizer_event.v1";
pub const OPTIMIZER_STATE_SLICE_SCHEMA_VERSION: &str = "optimizer_state_slice.v1";

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum OptimizerAlgorithm {
    Gepa,
    #[serde(rename = "go-ex")]
    GoEx,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerItemType {
    Run,
    Cursor,
    Candidate,
    FrontierCell,
    Rollout,
    ModelCall,
    Log,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum OptimizerStateSliceKind {
    Cursor,
    Candidates,
    Frontier,
    Board,
    Agents,
    Themes,
    DataEngine,
    Logs,
    Usage,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerLogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerItem {
    #[serde(rename = "type")]
    pub item_type: OptimizerItemType,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub raw: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerEvent {
    pub schema_version: String,
    #[serde(rename = "type")]
    pub event_type: String,
    pub sequence_number: u64,
    pub run_id: String,
    pub algorithm: OptimizerAlgorithm,
    pub created_at: String,
    #[serde(default)]
    pub item: Option<OptimizerItem>,
    #[serde(default)]
    pub delta: Map<String, Value>,
    #[serde(default)]
    pub error: Option<Value>,
    #[serde(default)]
    pub raw: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerStateSlice {
    pub schema_version: String,
    pub projection_schema_version: String,
    pub run_id: String,
    pub algorithm: OptimizerAlgorithm,
    pub slice: OptimizerStateSliceKind,
    pub cursor_seq: u64,
    pub updated_at: String,
    pub data: Value,
}
