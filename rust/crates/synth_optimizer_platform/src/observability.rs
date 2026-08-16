use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

pub const OPTIMIZER_EVENT_SCHEMA_VERSION: &str = "optimizer_event.v1";
pub const OPTIMIZER_STATE_SLICE_SCHEMA_VERSION: &str = "optimizer_state_slice.v1";
pub const LUNA_MED_POLICY_CONFIG: &str = "luna_med";
pub const SOL_MED_POLICY_CONFIG: &str = "sol_med";
pub const GEPA_PROPOSER_HARNESS: &str = "gepa_proposer";
pub const BANKING77_EVAL_HARNESS: &str = "banking77_eval";
pub const A3_TASK: &str = "banking77";
pub const OPTIMIZER_EVENT_SLOT: &str = "optimizer_run";
pub const PROPOSER_DELTA_EVENT_TYPE: &str = "proposer.delta";
pub const DEFAULT_PROPOSER_DELTA_CHANNEL: &str = "content";
pub const CHILD_ROLLOUT_ATTACHED_EVENT_TYPE: &str = "optimizer.child_rollout.attached";

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
    /// Missing sequence stays null. Never default to 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence_number: Option<u64>,
    pub run_id: String,
    #[serde(rename = "algorithm_id", alias = "algorithm")]
    pub algorithm: OptimizerAlgorithm,
    #[serde(default = "default_optimizer_event_slot")]
    pub slot: String,
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

impl OptimizerEvent {
    pub fn from_gepa_stream(
        sequence_number: u64,
        event_type: &str,
        message: &str,
        timestamp: impl Into<String>,
        fields: Value,
        run_id: impl Into<String>,
        algorithm: OptimizerAlgorithm,
    ) -> Self {
        let delta = if event_type == PROPOSER_DELTA_EVENT_TYPE {
            proposer_delta_fields_from_value(&fields)
        } else {
            let mut delta = fields.as_object().cloned().unwrap_or_default();
            if !message.is_empty() {
                delta
                    .entry("message".to_string())
                    .or_insert_with(|| Value::String(message.to_string()));
            }
            delta
        };
        Self {
            schema_version: OPTIMIZER_EVENT_SCHEMA_VERSION.into(),
            event_type: event_type.to_string(),
            sequence_number: Some(sequence_number),
            created_at: timestamp.into(),
            run_id: run_id.into(),
            algorithm,
            slot: OPTIMIZER_EVENT_SLOT.into(),
            item: None,
            delta,
            error: None,
            raw: serde_json::json!({
                "schema_version": "event_stream_record.v1",
                "event_type": event_type,
                "message": message,
                "fields": fields,
            }),
        }
    }
}

fn default_optimizer_event_slot() -> String {
    OPTIMIZER_EVENT_SLOT.to_string()
}

pub fn optimizer_event_log_id(optimizer_run_id: &str) -> String {
    format!("{OPTIMIZER_EVENT_SCHEMA_VERSION}:{optimizer_run_id}")
}

pub fn policy_ref(harness: &str, config: &str, code: Option<&str>) -> Value {
    let mut object = serde_json::Map::new();
    object.insert("harness".into(), Value::String(harness.to_string()));
    object.insert("config".into(), Value::String(config.to_string()));
    if let Some(code) = code {
        object.insert("code".into(), Value::String(code.to_string()));
    }
    Value::Object(object)
}

pub fn gepa_proposer_policy_ref(config: &str) -> Value {
    policy_ref(GEPA_PROPOSER_HARNESS, config, None)
}

pub fn container_child_eval_ref(rollout_id: &str, stream_id: &str, reward_url: &str) -> Value {
    serde_json::json!({
        "schema": "synth.resource-ref.v1",
        "kind": "container_rollout",
        "id": rollout_id,
        "role": "candidate_evaluation",
        "attributes": {
            "stream_id": stream_id,
            "reward_url": reward_url,
        }
    })
}

/// Workshop folds `proposer.delta` in place like `span.policy.data`.
/// Channel defaults to `content` when omitted; generation 0 is a real first generation.
pub fn proposer_delta_fields(generation: u64, channel: &str, text: &str) -> Map<String, Value> {
    let channel = if channel.trim().is_empty() {
        DEFAULT_PROPOSER_DELTA_CHANNEL
    } else {
        channel
    };
    let mut delta = Map::new();
    delta.insert("generation".into(), json!(generation));
    delta.insert("channel".into(), json!(channel));
    delta.insert("text".into(), json!(text));
    delta
}

fn proposer_delta_fields_from_value(fields: &Value) -> Map<String, Value> {
    let generation = fields.get("generation").and_then(Value::as_u64);
    let channel = fields
        .get("channel")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_PROPOSER_DELTA_CHANNEL);
    let text = fields.get("text").and_then(Value::as_str).unwrap_or("");
    let mut delta = proposer_delta_fields(generation.unwrap_or(0), channel, text);
    // Generation 0 is a valid first GEPA generation. Missing generation stays
    // absent rather than being coerced — the emit helper always supplies it.
    if generation.is_none() {
        delta.remove("generation");
    }
    delta
}

/// Pull live proposer text chunks from a GEPA proposer response. Empty text is dropped.
pub fn proposer_delta_chunks_from_response(response: &Value) -> Vec<(String, String)> {
    if let Some(chunks) = response
        .get("proposer_stream_chunks")
        .and_then(Value::as_array)
    {
        let extracted = chunks
            .iter()
            .filter_map(proposer_delta_chunk_from_value)
            .collect::<Vec<_>>();
        if !extracted.is_empty() {
            return extracted;
        }
    }
    let protocol_messages = response
        .get("received_messages")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| {
            response
                .pointer("/protocol/received")
                .and_then(Value::as_array)
                .cloned()
        })
        .unwrap_or_default();
    let protocol_chunks = proposer_delta_chunks_from_protocol(&protocol_messages);
    if !protocol_chunks.is_empty() {
        return protocol_chunks;
    }
    if let Some(text) = response
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        return vec![(DEFAULT_PROPOSER_DELTA_CHANNEL.to_string(), text.to_string())];
    }
    Vec::new()
}

pub fn proposer_delta_chunks_from_protocol(messages: &[Value]) -> Vec<(String, String)> {
    messages
        .iter()
        .filter_map(proposer_delta_chunk_from_protocol_message)
        .filter(|(_, text)| !text.is_empty())
        .collect()
}

fn proposer_delta_chunk_from_value(value: &Value) -> Option<(String, String)> {
    let channel = value
        .get("channel")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_PROPOSER_DELTA_CHANNEL)
        .to_string();
    let text = value.get("text").and_then(Value::as_str)?.to_string();
    if text.is_empty() {
        return None;
    }
    Some((channel, text))
}

fn proposer_delta_chunk_from_protocol_message(message: &Value) -> Option<(String, String)> {
    let method = message
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let params = message.get("params").unwrap_or(message);
    let item_type = params
        .pointer("/item/type")
        .or_else(|| params.pointer("/msg/type"))
        .or_else(|| params.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let channel = proposer_delta_channel_for(&method, &item_type)?;
    let text = protocol_delta_text(params)?;
    Some((channel.to_string(), text))
}

fn proposer_delta_channel_for(method: &str, item_type: &str) -> Option<&'static str> {
    let blob = format!("{method} {item_type}");
    if blob.contains("reasoning") || blob.contains("think") {
        return Some("reasoning");
    }
    if blob.contains("delta")
        || blob.contains("agent_message")
        || blob.contains("agentmessage")
        || blob.contains("output_text")
        || blob.contains("outputtext")
    {
        return Some(DEFAULT_PROPOSER_DELTA_CHANNEL);
    }
    None
}

fn protocol_delta_text(params: &Value) -> Option<String> {
    const POINTERS: [&str; 8] = [
        "/delta/text",
        "/item/delta/text",
        "/msg/delta/text",
        "/delta",
        "/text",
        "/item/text",
        "/msg/delta",
        "/item/delta",
    ];
    for pointer in POINTERS {
        match params.pointer(pointer) {
            Some(Value::String(text)) if !text.is_empty() => return Some(text.clone()),
            _ => {}
        }
    }
    None
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerStateSlice {
    pub schema_version: String,
    pub projection_schema_version: String,
    pub run_id: String,
    pub algorithm: OptimizerAlgorithm,
    pub slice: OptimizerStateSliceKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor_seq: Option<u64>,
    pub updated_at: String,
    pub data: Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn missing_sequence_deserializes_as_none_not_zero() {
        let event: OptimizerEvent = serde_json::from_value(json!({
            "schema_version": OPTIMIZER_EVENT_SCHEMA_VERSION,
            "type": "optimizer.heartbeat",
            "run_id": "gepa_luna",
            "algorithm": "gepa",
            "created_at": "2026-08-12T00:00:00Z",
            "raw": {}
        }))
        .unwrap();
        assert_eq!(event.sequence_number, None);
        assert_eq!(event.slot, OPTIMIZER_EVENT_SLOT);
    }

    #[test]
    fn present_sequence_is_preserved() {
        let event: OptimizerEvent = serde_json::from_value(json!({
            "schema_version": OPTIMIZER_EVENT_SCHEMA_VERSION,
            "type": "optimizer.run.started",
            "sequence_number": 3,
            "run_id": "gepa_luna",
            "algorithm": "gepa",
            "created_at": "2026-08-12T00:00:00Z",
            "raw": {}
        }))
        .unwrap();
        assert_eq!(event.sequence_number, Some(3));
    }

    #[test]
    fn child_eval_ref_is_resource_ref_not_signed_blob() {
        let ref_value =
            container_child_eval_ref("roll_abc", "stream_abc", "/reward?rollout_id=roll_abc");
        assert_eq!(ref_value["kind"], "container_rollout");
        assert_eq!(ref_value["id"], "roll_abc");
        assert_eq!(ref_value["attributes"]["stream_id"], "stream_abc");
        assert_eq!(
            ref_value["attributes"]["reward_url"],
            "/reward?rollout_id=roll_abc"
        );
        assert!(ref_value.get("frame").is_none());
        assert!(!ref_value.to_string().to_ascii_lowercase().contains("nev"));
        assert!(!ref_value.to_string().contains("child_eval_ref"));
    }

    #[test]
    fn luna_and_sol_are_policy_configs_not_tasks() {
        let luna = gepa_proposer_policy_ref(LUNA_MED_POLICY_CONFIG);
        let sol = gepa_proposer_policy_ref(SOL_MED_POLICY_CONFIG);
        assert_eq!(luna["harness"], GEPA_PROPOSER_HARNESS);
        assert_eq!(luna["config"], LUNA_MED_POLICY_CONFIG);
        assert_eq!(sol["config"], SOL_MED_POLICY_CONFIG);
        assert!(luna.get("harness_ref").is_none());
        assert_ne!(luna, sol);
        assert_eq!(A3_TASK, "banking77");
        assert_ne!(
            optimizer_event_log_id("gepa_luna"),
            optimizer_event_log_id("gepa_sol")
        );
    }

    #[test]
    fn proposer_delta_delta_is_generation_channel_text() {
        let event = OptimizerEvent::from_gepa_stream(
            4,
            PROPOSER_DELTA_EVENT_TYPE,
            "ignored message must not land in delta",
            "2026-08-12T00:00:00Z",
            json!({"generation": 2, "channel": "reasoning", "text": "chunk ", "extra": true}),
            "gepa_luna",
            OptimizerAlgorithm::Gepa,
        );
        assert_eq!(event.event_type, PROPOSER_DELTA_EVENT_TYPE);
        assert_eq!(event.sequence_number, Some(4));
        assert_eq!(event.delta.get("generation"), Some(&json!(2)));
        assert_eq!(event.delta.get("channel"), Some(&json!("reasoning")));
        assert_eq!(event.delta.get("text"), Some(&json!("chunk ")));
        assert!(event.delta.get("message").is_none());
        assert!(event.delta.get("extra").is_none());
    }

    #[test]
    fn proposer_delta_missing_generation_is_omitted_not_zero() {
        let event = OptimizerEvent::from_gepa_stream(
            1,
            PROPOSER_DELTA_EVENT_TYPE,
            "",
            "2026-08-12T00:00:00Z",
            json!({"channel": "content", "text": "hello"}),
            "gepa_luna",
            OptimizerAlgorithm::Gepa,
        );
        assert!(event.delta.get("generation").is_none());
        assert_eq!(event.delta.get("text"), Some(&json!("hello")));
    }

    #[test]
    fn protocol_item_deltas_become_channel_text_chunks() {
        let chunks = proposer_delta_chunks_from_protocol(&[
            json!({"method": "item/agentMessageDelta", "params": {"delta": "Hello "}}),
            json!({"method": "item/reasoning/delta", "params": {"delta": {"text": "think"}}}),
            json!({"method": "turn/completed", "params": {}}),
        ]);
        assert_eq!(
            chunks,
            vec![
                ("content".to_string(), "Hello ".to_string()),
                ("reasoning".to_string(), "think".to_string()),
            ]
        );
    }
}
