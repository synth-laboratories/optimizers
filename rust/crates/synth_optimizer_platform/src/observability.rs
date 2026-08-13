use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fmt;

pub const OPTIMIZER_EVENT_SCHEMA_VERSION: &str = "optimizer_event.v1";
pub const OPTIMIZER_STATE_SLICE_SCHEMA_VERSION: &str = "optimizer_state_slice.v1";
pub const OPTIMIZER_RUN_SCHEMA_VERSION: &str = "optimizer_run.v1";

/// Known algorithm id constants. Unknown ids remain valid as plain strings.
pub mod algorithm_ids {
    pub const GEPA: &str = "gepa";
    pub const GO_EX: &str = "go-ex";
    pub const SFT: &str = "sft";
}

pub mod slice_ids {
    pub const RUN_SUMMARY: &str = "run.summary";
    pub const RUN_TIMELINE: &str = "run.timeline";
    pub const RUN_USAGE: &str = "run.usage";
    pub const RUN_LOGS: &str = "run.logs";
    pub const RUN_ARTIFACTS: &str = "run.artifacts";
    pub const RUN_EXECUTION: &str = "run.execution";
    pub const GEPA_CANDIDATES: &str = "gepa.candidates";
    pub const GEPA_FRONTIER: &str = "gepa.frontier";
    pub const GEPA_REFLECTIONS: &str = "gepa.reflections";
    pub const GOEX_BOARD: &str = "go-ex.board";
    pub const GOEX_THEMES: &str = "go-ex.themes";
    pub const GOEX_DATA_ENGINE: &str = "go-ex.data_engine";
    pub const SFT_TRAINING_CURVES: &str = "sft.training_curves";
    pub const SFT_CHECKPOINTS: &str = "sft.checkpoints";
    pub const SFT_CHECKPOINT_EVALUATIONS: &str = "sft.checkpoint_evaluations";
    pub const SFT_DATASET: &str = "sft.dataset";
    pub const SFT_COMPUTE: &str = "sft.compute";
    pub const SFT_EXAMPLES: &str = "sft.examples";
}

pub mod item_kinds {
    pub const RUN: &str = "run";
    pub const CURSOR: &str = "cursor";
    pub const CANDIDATE: &str = "candidate";
    pub const FRONTIER_CELL: &str = "frontier_cell";
    pub const ROLLOUT: &str = "rollout";
    pub const MODEL_CALL: &str = "model_call";
    pub const LOG: &str = "log";
    pub const CHECKPOINT: &str = "checkpoint";
    pub const METRIC: &str = "metric";
    pub const EVALUATION: &str = "evaluation";
    pub const DATASET: &str = "dataset";
    pub const ARTIFACT: &str = "artifact";
    pub const PROVIDER_OPERATION: &str = "provider_operation";
}

/// Validated algorithm id string. Unknown future algorithms are preserved, not rejected.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(transparent)]
pub struct AlgorithmId(String);

impl AlgorithmId {
    pub fn new(value: impl Into<String>) -> Result<Self, String> {
        let value = value.into().trim().to_string();
        if value.is_empty() {
            return Err("algorithm_id must be non-empty".into());
        }
        if value.len() > 64 {
            return Err("algorithm_id exceeds 64 characters".into());
        }
        if !value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
        {
            return Err("algorithm_id contains invalid characters".into());
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn gepa() -> Self {
        Self(algorithm_ids::GEPA.into())
    }

    pub fn go_ex() -> Self {
        Self(algorithm_ids::GO_EX.into())
    }

    pub fn sft() -> Self {
        Self(algorithm_ids::SFT.into())
    }

    pub fn is_known(&self) -> bool {
        matches!(
            self.0.as_str(),
            algorithm_ids::GEPA | algorithm_ids::GO_EX | algorithm_ids::SFT
        )
    }
}

impl fmt::Display for AlgorithmId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for AlgorithmId {
    fn from(value: &str) -> Self {
        Self::new(value).unwrap_or_else(|_| Self("unknown".into()))
    }
}

/// Legacy closed enum retained for dual-read of deployed consumers.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum OptimizerAlgorithm {
    Gepa,
    #[serde(rename = "go-ex")]
    GoEx,
}

impl OptimizerAlgorithm {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gepa => algorithm_ids::GEPA,
            Self::GoEx => algorithm_ids::GO_EX,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            algorithm_ids::GEPA => Some(Self::Gepa),
            algorithm_ids::GO_EX => Some(Self::GoEx),
            _ => None,
        }
    }
}

/// Legacy closed item enum. Prefer [`item_kinds`] strings via [`OptimizerItem::kind`].
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
    Checkpoint,
    Metric,
    Evaluation,
    Dataset,
    Artifact,
    ProviderOperation,
}

impl OptimizerItemType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Run => item_kinds::RUN,
            Self::Cursor => item_kinds::CURSOR,
            Self::Candidate => item_kinds::CANDIDATE,
            Self::FrontierCell => item_kinds::FRONTIER_CELL,
            Self::Rollout => item_kinds::ROLLOUT,
            Self::ModelCall => item_kinds::MODEL_CALL,
            Self::Log => item_kinds::LOG,
            Self::Checkpoint => item_kinds::CHECKPOINT,
            Self::Metric => item_kinds::METRIC,
            Self::Evaluation => item_kinds::EVALUATION,
            Self::Dataset => item_kinds::DATASET,
            Self::Artifact => item_kinds::ARTIFACT,
            Self::ProviderOperation => item_kinds::PROVIDER_OPERATION,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            item_kinds::RUN => Some(Self::Run),
            item_kinds::CURSOR => Some(Self::Cursor),
            item_kinds::CANDIDATE => Some(Self::Candidate),
            item_kinds::FRONTIER_CELL => Some(Self::FrontierCell),
            item_kinds::ROLLOUT => Some(Self::Rollout),
            item_kinds::MODEL_CALL => Some(Self::ModelCall),
            item_kinds::LOG => Some(Self::Log),
            item_kinds::CHECKPOINT => Some(Self::Checkpoint),
            item_kinds::METRIC => Some(Self::Metric),
            item_kinds::EVALUATION => Some(Self::Evaluation),
            item_kinds::DATASET => Some(Self::Dataset),
            item_kinds::ARTIFACT => Some(Self::Artifact),
            item_kinds::PROVIDER_OPERATION => Some(Self::ProviderOperation),
            _ => None,
        }
    }
}

/// Legacy closed slice enum. Prefer namespaced [`slice_ids`].
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

impl OptimizerStateSliceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cursor => "cursor",
            Self::Candidates => slice_ids::GEPA_CANDIDATES,
            Self::Frontier => slice_ids::GEPA_FRONTIER,
            Self::Board => slice_ids::GOEX_BOARD,
            Self::Agents => "agents",
            Self::Themes => slice_ids::GOEX_THEMES,
            Self::DataEngine => slice_ids::GOEX_DATA_ENGINE,
            Self::Logs => slice_ids::RUN_LOGS,
            Self::Usage => slice_ids::RUN_USAGE,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerLogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct OptimizerCapabilities {
    #[serde(default)]
    pub cancel: bool,
    #[serde(default)]
    pub pause: bool,
    #[serde(default)]
    pub resume: bool,
    #[serde(default)]
    pub stream_events: bool,
    #[serde(default)]
    pub state_slices: bool,
    #[serde(default)]
    pub candidates: bool,
    #[serde(default)]
    pub checkpoints: bool,
    #[serde(default)]
    pub checkpoint_evaluations: bool,
    #[serde(default)]
    pub inference_endpoint: bool,
    #[serde(default)]
    pub local_slot_binding: bool,
}

impl OptimizerCapabilities {
    pub fn gepa_defaults() -> Self {
        Self {
            cancel: true,
            pause: true,
            resume: true,
            stream_events: true,
            state_slices: true,
            candidates: true,
            checkpoints: false,
            checkpoint_evaluations: false,
            inference_endpoint: false,
            local_slot_binding: false,
        }
    }

    pub fn goex_defaults() -> Self {
        Self {
            cancel: true,
            pause: true,
            resume: true,
            stream_events: true,
            state_slices: true,
            candidates: true,
            checkpoints: true,
            checkpoint_evaluations: true,
            inference_endpoint: false,
            local_slot_binding: true,
        }
    }

    pub fn sft_defaults() -> Self {
        Self {
            cancel: true,
            pause: true,
            resume: true,
            stream_events: true,
            state_slices: true,
            candidates: false,
            checkpoints: true,
            checkpoint_evaluations: true,
            inference_endpoint: true,
            local_slot_binding: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerItem {
    /// Dual-read: prefer `kind` string; fall back to closed `type` enum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub item_type: Option<OptimizerItemType>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub raw: Value,
}

impl OptimizerItem {
    pub fn kind_id(&self) -> &str {
        if let Some(kind) = self.kind.as_deref() {
            return kind;
        }
        self.item_type
            .map(OptimizerItemType::as_str)
            .unwrap_or("unknown")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerEvent {
    pub schema_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
    #[serde(rename = "type")]
    pub event_type: String,
    pub sequence_number: u64,
    #[serde(default, alias = "occurred_at")]
    pub created_at: String,
    #[serde(alias = "optimizer_run_id")]
    pub run_id: String,
    /// Preferred forward-compatible algorithm id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm_id: Option<String>,
    /// Legacy closed algorithm field retained for dual-read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<OptimizerAlgorithm>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<OptimizerLogLevel>,
    #[serde(default)]
    pub item: Option<OptimizerItem>,
    #[serde(default)]
    pub delta: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage_delta: Option<Map<String, Value>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifact_refs: Vec<Value>,
    #[serde(default)]
    pub error: Option<Value>,
    #[serde(default)]
    pub raw: Value,
}

impl OptimizerEvent {
    pub fn algorithm_id(&self) -> String {
        if let Some(id) = self.algorithm_id.as_ref().filter(|s| !s.is_empty()) {
            return id.clone();
        }
        self.algorithm
            .map(OptimizerAlgorithm::as_str)
            .unwrap_or("unknown")
            .to_string()
    }

    /// Build a canonical envelope from a GEPA/OSS stream record.
    pub fn from_gepa_stream(
        sequence_number: u64,
        event_type: &str,
        message: &str,
        timestamp: impl Into<String>,
        fields: Value,
        run_id: impl Into<String>,
        algorithm_id: impl Into<String>,
    ) -> Self {
        let run_id = run_id.into();
        let algorithm_id = algorithm_id.into();
        let algorithm = OptimizerAlgorithm::parse(&algorithm_id);
        let mut delta = fields.as_object().cloned().unwrap_or_default();
        if !message.is_empty() {
            delta
                .entry("message".to_string())
                .or_insert_with(|| Value::String(message.to_string()));
        }
        let item = infer_item_from_event_type(event_type, &fields);
        Self {
            schema_version: OPTIMIZER_EVENT_SCHEMA_VERSION.into(),
            event_id: Some(format!("{run_id}:{sequence_number}")),
            event_type: event_type.to_string(),
            sequence_number,
            created_at: timestamp.into(),
            run_id: run_id.clone(),
            algorithm_id: Some(algorithm_id),
            algorithm,
            level: Some(OptimizerLogLevel::Info),
            item,
            delta,
            snapshot: None,
            usage_delta: extract_usage_delta(&fields),
            artifact_refs: Vec::new(),
            error: None,
            raw: serde_json::json!({
                "schema_version": "event_stream_record.v1",
                "event_type": event_type,
                "message": message,
                "fields": fields,
            }),
        }
    }

    /// Build a canonical envelope from a Go-Ex / GELO native event line.
    pub fn from_goex_event(
        sequence_number: u64,
        event_type: &str,
        occurred_at: impl Into<String>,
        payload: Value,
        run_id: impl Into<String>,
    ) -> Self {
        let run_id = run_id.into();
        let delta = payload.as_object().cloned().unwrap_or_default();
        let item = infer_item_from_event_type(event_type, &payload);
        Self {
            schema_version: OPTIMIZER_EVENT_SCHEMA_VERSION.into(),
            event_id: Some(format!("{run_id}:{sequence_number}")),
            event_type: event_type.to_string(),
            sequence_number,
            created_at: occurred_at.into(),
            run_id,
            algorithm_id: Some(algorithm_ids::GO_EX.into()),
            algorithm: Some(OptimizerAlgorithm::GoEx),
            level: Some(OptimizerLogLevel::Info),
            item,
            delta,
            snapshot: None,
            usage_delta: extract_usage_delta(&payload),
            artifact_refs: Vec::new(),
            error: payload.get("error").cloned(),
            raw: serde_json::json!({
                "schema_version": "goex_event.v1",
                "event_type": event_type,
                "payload": payload,
            }),
        }
    }
}

fn infer_item_from_event_type(event_type: &str, fields: &Value) -> Option<OptimizerItem> {
    let lower = event_type.to_ascii_lowercase();
    let (kind, id_keys) = if lower.contains("candidate") {
        (item_kinds::CANDIDATE, &["candidate_id", "id"][..])
    } else if lower.contains("frontier") {
        (item_kinds::FRONTIER_CELL, &["cell_id", "id"][..])
    } else if lower.contains("checkpoint") {
        (item_kinds::CHECKPOINT, &["checkpoint_id", "id"][..])
    } else if lower.contains("rollout") {
        (item_kinds::ROLLOUT, &["rollout_id", "id"][..])
    } else if lower.contains("theme") {
        (item_kinds::METRIC, &["theme_id", "theme", "id"][..])
    } else {
        return None;
    };
    let id = id_keys
        .iter()
        .find_map(|key| fields.get(*key).and_then(Value::as_str))
        .map(str::to_string);
    Some(OptimizerItem {
        kind: Some(kind.into()),
        item_type: None,
        id,
        status: fields
            .get("status")
            .and_then(Value::as_str)
            .map(str::to_string),
        raw: fields.clone(),
    })
}

fn extract_usage_delta(fields: &Value) -> Option<Map<String, Value>> {
    let mut out = Map::new();
    for key in [
        "cost_usd",
        "prompt_tokens",
        "completion_tokens",
        "rollouts",
        "wall_time_ms",
    ] {
        if let Some(value) = fields.get(key) {
            out.insert(key.into(), value.clone());
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerStateSlice {
    pub schema_version: String,
    pub projection_schema_version: String,
    #[serde(alias = "optimizer_run_id")]
    pub run_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<OptimizerAlgorithm>,
    /// Preferred namespaced slice id (e.g. `gepa.frontier`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slice_id: Option<String>,
    /// Legacy closed slice enum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slice: Option<OptimizerStateSliceKind>,
    pub cursor_seq: u64,
    pub updated_at: String,
    pub data: Value,
}

impl OptimizerStateSlice {
    pub fn algorithm_id(&self) -> String {
        if let Some(id) = self.algorithm_id.as_ref().filter(|s| !s.is_empty()) {
            return id.clone();
        }
        self.algorithm
            .map(OptimizerAlgorithm::as_str)
            .unwrap_or("unknown")
            .to_string()
    }

    pub fn slice_id(&self) -> String {
        if let Some(id) = self.slice_id.as_ref().filter(|s| !s.is_empty()) {
            return id.clone();
        }
        self.slice
            .map(OptimizerStateSliceKind::as_str)
            .unwrap_or("unknown")
            .to_string()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OptimizerExecutionBinding {
    pub kind: String,
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OptimizerResourceRef {
    pub kind: String,
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct OptimizerUsageSummary {
    #[serde(default)]
    pub cost_usd: f64,
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub rollouts: u64,
    #[serde(default)]
    pub wall_time_ms: u64,
    #[serde(default)]
    pub extra: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OptimizerRunRecord {
    pub schema_version: String,
    pub id: String,
    pub algorithm_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub algorithm_version: Option<String>,
    pub status: String,
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_ref: Option<String>,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<String>,
    #[serde(default)]
    pub cursor_seq: u64,
    #[serde(default)]
    pub capabilities: OptimizerCapabilities,
    #[serde(default)]
    pub execution_bindings: Vec<OptimizerExecutionBinding>,
    #[serde(default)]
    pub input_refs: Vec<OptimizerResourceRef>,
    #[serde(default)]
    pub output_refs: Vec<OptimizerResourceRef>,
    #[serde(default)]
    pub visual_refs: Vec<OptimizerResourceRef>,
    #[serde(default)]
    pub summary: Value,
    #[serde(default)]
    pub usage: OptimizerUsageSummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OptimizerRelationship {
    pub from_kind: String,
    pub from_id: String,
    pub edge: String,
    pub to_kind: String,
    pub to_id: String,
    #[serde(default)]
    pub metadata: Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn algorithm_id_accepts_unknown_future_ids() {
        let id = AlgorithmId::new("future.algo-v2").unwrap();
        assert_eq!(id.as_str(), "future.algo-v2");
        assert!(!id.is_known());
    }

    #[test]
    fn dual_reads_legacy_algorithm_enum() {
        let event: OptimizerEvent = serde_json::from_value(json!({
            "schema_version": OPTIMIZER_EVENT_SCHEMA_VERSION,
            "type": "gepa.run.started",
            "sequence_number": 1,
            "run_id": "opt_1",
            "algorithm": "gepa",
            "created_at": "2026-08-09T00:00:00Z",
            "raw": {}
        }))
        .unwrap();
        assert_eq!(event.algorithm_id(), "gepa");
    }

    #[test]
    fn preserves_unknown_algorithm_id_without_enum() {
        let event: OptimizerEvent = serde_json::from_value(json!({
            "schema_version": OPTIMIZER_EVENT_SCHEMA_VERSION,
            "type": "sft.training.started",
            "sequence_number": 2,
            "optimizer_run_id": "opt_2",
            "algorithm_id": "sft",
            "created_at": "2026-08-09T00:00:01Z",
            "item": {"kind": "checkpoint", "id": "ckpt_1", "raw": {}},
            "raw": {}
        }))
        .unwrap();
        assert_eq!(event.algorithm_id(), "sft");
        assert_eq!(event.item.as_ref().unwrap().kind_id(), "checkpoint");
    }

    #[test]
    fn from_gepa_stream_builds_canonical_envelope() {
        let event = OptimizerEvent::from_gepa_stream(
            3,
            "candidate.accepted",
            "accepted",
            "2026-08-09T15:00:00Z",
            json!({"run_id": "gepa_1", "candidate_id": "c1", "cost_usd": 0.2}),
            "gepa_1",
            "gepa",
        );
        assert_eq!(event.schema_version, OPTIMIZER_EVENT_SCHEMA_VERSION);
        assert_eq!(event.algorithm_id(), "gepa");
        assert_eq!(event.sequence_number, 3);
        assert_eq!(event.item.as_ref().unwrap().id.as_deref(), Some("c1"));
        assert_eq!(
            event.usage_delta.as_ref().unwrap().get("cost_usd").unwrap(),
            &json!(0.2)
        );
    }

    #[test]
    fn from_goex_event_builds_canonical_envelope() {
        let event = OptimizerEvent::from_goex_event(
            4,
            "theme.updated",
            "2026-08-09T15:01:00Z",
            json!({"theme": "oak"}),
            "goex_1",
        );
        assert_eq!(event.algorithm_id(), "go-ex");
        assert_eq!(event.event_type, "theme.updated");
        assert_eq!(event.delta.get("theme").unwrap(), "oak");
    }

    #[test]
    fn state_slice_prefers_namespaced_slice_id() {
        let slice: OptimizerStateSlice = serde_json::from_value(json!({
            "schema_version": OPTIMIZER_STATE_SLICE_SCHEMA_VERSION,
            "projection_schema_version": "gepa.frontier.v1",
            "run_id": "opt_1",
            "algorithm_id": "gepa",
            "slice_id": "gepa.frontier",
            "cursor_seq": 12,
            "updated_at": "2026-08-09T00:00:00Z",
            "data": {"cells": []}
        }))
        .unwrap();
        assert_eq!(slice.slice_id(), "gepa.frontier");
    }

    #[test]
    fn optimizer_run_record_round_trips() {
        let run = OptimizerRunRecord {
            schema_version: OPTIMIZER_RUN_SCHEMA_VERSION.into(),
            id: "opt_gepa_fixture".into(),
            algorithm_id: algorithm_ids::GEPA.into(),
            algorithm_version: Some("1.0.0".into()),
            status: "running".into(),
            source: "local".into(),
            objective: Some("maximize train reward".into()),
            project_ref: None,
            session_ref: Some("session_1".into()),
            created_at: "2026-08-09T12:00:00Z".into(),
            started_at: Some("2026-08-09T12:00:01Z".into()),
            finished_at: None,
            cursor_seq: 8,
            capabilities: OptimizerCapabilities::gepa_defaults(),
            execution_bindings: vec![],
            input_refs: vec![],
            output_refs: vec![],
            visual_refs: vec![],
            summary: json!({"bestScore": 0.82}),
            usage: OptimizerUsageSummary {
                cost_usd: 2.14,
                prompt_tokens: 12000,
                completion_tokens: 4000,
                rollouts: 48,
                wall_time_ms: 90_000,
                extra: Map::new(),
            },
            error: None,
        };
        let value = serde_json::to_value(&run).unwrap();
        let back: OptimizerRunRecord = serde_json::from_value(value).unwrap();
        assert_eq!(back.id, run.id);
        assert!(back.capabilities.candidates);
    }
}
