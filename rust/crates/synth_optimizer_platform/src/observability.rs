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

// ---------------------------------------------------------------------------
// Event vocabulary (P0-5)
//
// Two feeds leave this repo, and Workshop matches string literals against both:
//
//   `optimizer_event.v1`     — the canonical per-run spool written by
//                              `EventStream::emit` (events.optimizer.jsonl).
//   `service_run_events.v1`  — the projection served by `GET /runs/{id}/events`
//                              (`public_event_kind` in synth_gepa/src/service.rs).
//
// The two constants below are the declared vocabulary. `mod vocabulary` scans
// the workspace sources and fails if a source can emit a name the constants do
// not declare (or declares a name nothing can emit), and fails if the committed
// `contracts/event_vocabulary.json` disagrees. Nothing here adds an emitter:
// a name Workshop matches that is absent from these lists has no producer.
// ---------------------------------------------------------------------------

/// Feed id for the canonical per-run `optimizer_event.v1` spool.
pub const OPTIMIZER_EVENT_FEED: &str = "optimizer_event.v1";
/// Feed id for the Workshop-facing projection on `GET /runs/{id}/events`.
pub const SERVICE_RUN_EVENTS_FEED: &str = "service_run_events.v1";

pub const GEPA_RUN_CANCELLED_EVENT_TYPE: &str = "gepa.run.cancelled";
pub const GEPA_RUN_FAILED_EVENT_TYPE: &str = "gepa.run.failed";

/// Every event-type constant in this module, keyed by its Rust identifier.
/// The vocabulary scan resolves emit sites that pass a constant rather than a
/// literal through this table.
pub const EVENT_TYPE_CONSTANTS: &[(&str, &str)] = &[
    (
        "CHILD_ROLLOUT_ATTACHED_EVENT_TYPE",
        CHILD_ROLLOUT_ATTACHED_EVENT_TYPE,
    ),
    (
        "GEPA_RUN_CANCELLED_EVENT_TYPE",
        GEPA_RUN_CANCELLED_EVENT_TYPE,
    ),
    ("GEPA_RUN_FAILED_EVENT_TYPE", GEPA_RUN_FAILED_EVENT_TYPE),
    ("PROPOSER_DELTA_EVENT_TYPE", PROPOSER_DELTA_EVENT_TYPE),
];

/// Emit sites whose event type is a local binding rather than a literal or a
/// constant, with the constants that binding can hold. The vocabulary scan
/// fails on any emit site that is none of the three, so a new indirect emitter
/// cannot land without declaring what it emits.
pub const INDIRECT_EMIT_BINDINGS: &[(&str, &[&str])] = &[(
    "terminal_event_type",
    &[
        "GEPA_RUN_CANCELLED_EVENT_TYPE",
        "GEPA_RUN_FAILED_EVENT_TYPE",
    ],
)];

/// Complete, sorted set of event types the `optimizer_event.v1` feed can carry.
pub const OPTIMIZER_EVENT_TYPES: &[&str] = &[
    "candidate.accepted",
    "candidate.deferred",
    "candidate.duplicate_skipped",
    "candidate.evaluated",
    "candidate.full_train_evaluated",
    "candidate.leakage_detected",
    "candidate.minibatch_evaluated",
    "candidate.registered",
    "candidate.rejected",
    "container.contract.verified",
    "container.program.loaded",
    "container.task_info.loaded",
    "container.task_info.missing",
    "frontier.snapshot",
    "frontier.updated",
    "gepa.run.cancelled",
    "gepa.run.failed",
    "gepa.run.finished",
    "gepa.run.started",
    "gepa.stop",
    "heldout.blocked",
    "heldout.completed",
    "heldout.partial",
    "heldout.skipped",
    "objective_set.declared",
    "optimizer.candidate_evaluation.allocated",
    "optimizer.candidate_evaluation.attempt.failed",
    "optimizer.child_rollout.attached",
    "optimizer.evaluation.coverage.updated",
    "optimizer.evaluation_result.received",
    "optimizer.limit.estimate_updated",
    "optimizer.rollout_queue.updated",
    "optimizer.state.transitioned",
    "parent_minibatch_reference.completed",
    "pipeline.speculative_release.enqueued",
    "pipeline.speculative_tail.discarded",
    "pipeline.stage_workers.adjusted",
    "pipeline.stale_item.discarded",
    "pipeline.stale_item.patched",
    "pipeline.stale_item.reviewed",
    "proposer.completed",
    "proposer.delta",
    "proposer.started",
    "rollout.attempt.failed",
    "rollout.chunk.finished",
    "rollout.chunk.started",
    "rollout.circuit_breaker.tripped",
    "rollout.concurrency.adjusted",
    "rollout.failure_rate.updated",
    "rollout.outcome.duplicate_ignored",
    "rollout.stale_skipped",
    "runtime.job.completed",
    "runtime.throughput.warning",
    "score_chart.written",
    "storage.snapshot.recorded",
    "taskset.tasks.loaded",
    "workspace.persisted",
];

/// Complete, sorted set of kinds `GET /runs/{id}/events` can project.
pub const SERVICE_RUN_EVENT_KINDS: &[&str] = &[
    "candidate.accepted",
    "candidate.rejected",
    "candidate.scored",
    "frontier.updated",
    "generation.started",
    "heldout.completed",
    "heldout.started",
    "proposer.completed",
    "run.status_changed",
    "run.terminal",
    "usage.tick",
];

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

/// P0-5 lock. Scans the workspace sources for everything that can reach a feed
/// and diffs it against the declared constants and the committed
/// `contracts/event_vocabulary.json`. Adding an emitter, renaming one, or
/// hand-editing the JSON fails here in under a second.
#[cfg(test)]
mod vocabulary {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::{Path, PathBuf};

    fn repo_root() -> PathBuf {
        // .../rust/crates/synth_optimizer_platform -> repo root
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .expect("repo root above rust/crates/<crate>")
            .to_path_buf()
    }

    fn rust_sources() -> Vec<PathBuf> {
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if path.file_name().is_some_and(|name| name == "target") {
                        continue;
                    }
                    walk(&path, out);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    out.push(path);
                }
            }
        }
        let mut out = Vec::new();
        walk(&repo_root().join("rust"), &mut out);
        out.sort();
        out
    }

    /// One Rust source split into code bytes and literal/comment bytes.
    ///
    /// Every question the scan asks — is this `.emit(` real code, where does
    /// this argument end, what strings does it contain — is wrong without this.
    /// The scanner's own source contains `.emit(`, `{` and `}` inside string and
    /// char literals, so a naive scan mis-reads itself first.
    struct Lexed {
        source: String,
        /// `true` for bytes that are code (outside comments and literals).
        is_code: Vec<bool>,
        /// Byte ranges of string-literal *contents*, in order.
        strings: Vec<(usize, usize)>,
    }

    impl Lexed {
        fn new(source: &str) -> Self {
            let bytes = source.as_bytes();
            let mut is_code = vec![true; bytes.len()];
            let mut strings = Vec::new();
            let mut index = 0usize;
            while index < bytes.len() {
                let rest = &source[index..];
                if rest.starts_with("//") {
                    let end = rest.find('\n').map_or(bytes.len(), |at| index + at);
                    is_code[index..end].fill(false);
                    index = end;
                } else if rest.starts_with("/*") {
                    let mut depth = 0usize;
                    let mut cursor = index;
                    while cursor < bytes.len() {
                        if source[cursor..].starts_with("/*") {
                            depth += 1;
                            cursor += 2;
                        } else if source[cursor..].starts_with("*/") {
                            depth -= 1;
                            cursor += 2;
                            if depth == 0 {
                                break;
                            }
                        } else {
                            cursor += 1;
                        }
                    }
                    is_code[index..cursor.min(bytes.len())].fill(false);
                    index = cursor;
                } else if let Some(hashes) = raw_string_hashes(source, index) {
                    let open = index + rest.find('"').expect("raw string opener") + 1;
                    let terminator = format!("\"{}", "#".repeat(hashes));
                    let close = source[open..]
                        .find(&terminator)
                        .map_or(bytes.len(), |at| open + at);
                    strings.push((open, close));
                    let end = (close + terminator.len()).min(bytes.len());
                    is_code[index..end].fill(false);
                    index = end;
                } else if bytes[index] == b'"' {
                    let open = index + 1;
                    let mut cursor = open;
                    while cursor < bytes.len() {
                        match bytes[cursor] {
                            b'\\' => cursor += 2,
                            b'"' => break,
                            _ => cursor += 1,
                        }
                    }
                    let close = cursor.min(bytes.len());
                    strings.push((open, close));
                    let end = (close + 1).min(bytes.len());
                    is_code[index..end].fill(false);
                    index = end;
                } else if bytes[index] == b'\'' {
                    match char_literal_end(source, index) {
                        // A char literal is not code; a lifetime is.
                        Some(end) => {
                            is_code[index..end].fill(false);
                            index = end;
                        }
                        None => index += 1,
                    }
                } else {
                    index += 1;
                }
            }
            Self {
                source: source.to_string(),
                is_code,
                strings,
            }
        }

        fn code_at(&self, index: usize) -> bool {
            self.is_code.get(index).copied().unwrap_or(false)
        }

        /// Offsets where `needle` appears as code.
        fn code_matches(&self, needle: &str) -> Vec<usize> {
            let mut out = Vec::new();
            let mut from = 0usize;
            while let Some(at) = self.source[from..].find(needle) {
                let start = from + at;
                if (start..start + needle.len()).all(|index| self.code_at(index)) {
                    out.push(start);
                }
                from = start + 1;
            }
            out
        }

        fn line_of(&self, index: usize) -> usize {
            self.source[..index].matches('\n').count() + 1
        }

        /// End of the item starting at `from`: past its balanced block, or past
        /// its terminating `;` when it has no block.
        fn item_end(&self, from: usize) -> Option<usize> {
            let mut depth = 0usize;
            for (offset, ch) in self.source[from..].char_indices() {
                let index = from + offset;
                if !self.code_at(index) {
                    continue;
                }
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(index + 1);
                        }
                    }
                    ';' if depth == 0 => return Some(index + 1),
                    _ => {}
                }
            }
            None
        }

        /// Byte ranges guarded by a `#[cfg(test)]` attribute.
        fn test_item_ranges(&self) -> Vec<(usize, usize)> {
            let mut out: Vec<(usize, usize)> = Vec::new();
            for start in self.code_matches("#[cfg(test)]") {
                if out.iter().any(|(from, to)| start >= *from && start < *to) {
                    continue;
                }
                if let Some(end) = self.item_end(start + "#[cfg(test)]".len()) {
                    out.push((start, end));
                }
            }
            out
        }

        /// End of the first argument of the call whose `(` is at `open`.
        fn first_argument_end(&self, open: usize) -> usize {
            let mut depth = 0i32;
            for (offset, ch) in self.source[open + 1..].char_indices() {
                let index = open + 1 + offset;
                if !self.code_at(index) {
                    continue;
                }
                match ch {
                    '(' | '[' | '{' => depth += 1,
                    ')' | ']' | '}' if depth == 0 => return index,
                    ')' | ']' | '}' => depth -= 1,
                    ',' if depth == 0 => return index,
                    _ => {}
                }
            }
            self.source.len()
        }

        fn strings_within(&self, from: usize, to: usize) -> Vec<&str> {
            self.strings
                .iter()
                .filter(|(start, end)| *start >= from && *end <= to)
                .map(|(start, end)| &self.source[*start..*end])
                .collect()
        }
    }

    fn raw_string_hashes(source: &str, index: usize) -> Option<usize> {
        let rest = source[index..].as_bytes();
        let mut cursor = 0usize;
        if rest.first() == Some(&b'b') {
            cursor += 1;
        }
        if rest.get(cursor) != Some(&b'r') {
            return None;
        }
        cursor += 1;
        let mut hashes = 0usize;
        while rest.get(cursor) == Some(&b'#') {
            hashes += 1;
            cursor += 1;
        }
        (rest.get(cursor) == Some(&b'"')).then_some(hashes)
    }

    /// `Some(end)` when the `'` at `index` opens a char literal, `None` for a lifetime.
    fn char_literal_end(source: &str, index: usize) -> Option<usize> {
        let bytes = source.as_bytes();
        if bytes.get(index + 1) == Some(&b'\\') {
            let mut cursor = index + 2;
            while cursor < bytes.len() && bytes[cursor] != b'\'' {
                cursor += 1;
            }
            return (cursor < bytes.len()).then_some(cursor + 1);
        }
        let mut chars = source[index + 1..].char_indices();
        let (_, first) = chars.next()?;
        let after = index + 1 + first.len_utf8();
        (bytes.get(after) == Some(&b'\'')).then_some(after + 1)
    }

    fn constants() -> BTreeMap<&'static str, &'static str> {
        EVENT_TYPE_CONSTANTS.iter().copied().collect()
    }

    /// Event types reachable through `EventStream::emit` anywhere in the workspace,
    /// excluding `#[cfg(test)]` items (a test emitter is not a producer).
    fn scanned_optimizer_event_types() -> BTreeSet<String> {
        let constants = constants();
        let bindings: BTreeMap<&str, &[&str]> = INDIRECT_EMIT_BINDINGS.iter().copied().collect();
        let mut found = BTreeSet::new();
        for path in rust_sources() {
            let raw = std::fs::read_to_string(&path).expect("read rust source");
            let lexed = Lexed::new(&raw);
            let test_ranges = lexed.test_item_ranges();
            for start in lexed.code_matches(".emit(") {
                if test_ranges
                    .iter()
                    .any(|(from, to)| start >= *from && start < *to)
                {
                    continue;
                }
                let open = start + ".emit".len();
                let end = lexed.first_argument_end(open);
                let expression = &lexed.source[open + 1..end];
                let literals = lexed.strings_within(open + 1, end);
                let mut resolved = literals.len();
                for literal in literals {
                    found.insert(literal.to_string());
                }
                for (name, value) in &constants {
                    if expression.contains(name) {
                        found.insert((*value).to_string());
                        resolved += 1;
                    }
                }
                for (binding, names) in &bindings {
                    if expression
                        .split(|ch: char| !ch.is_alphanumeric() && ch != '_')
                        .any(|token| token == *binding)
                    {
                        for name in names.iter() {
                            let value = constants
                                .get(name)
                                .unwrap_or_else(|| panic!("{name} is not in EVENT_TYPE_CONSTANTS"));
                            found.insert((*value).to_string());
                        }
                        resolved += 1;
                    }
                }
                assert!(
                    resolved > 0,
                    "{}:{}: emit() event type `{}` is neither a literal, an \
                     EVENT_TYPE_CONSTANTS entry, nor an INDIRECT_EMIT_BINDINGS entry. \
                     Declare what it emits in observability.rs.",
                    path.display(),
                    lexed.line_of(start),
                    expression.trim()
                );
            }
        }
        found
    }

    /// Kinds `GET /runs/{id}/events` can project, read out of `public_event_kind`.
    fn scanned_service_run_event_kinds() -> BTreeSet<String> {
        let path = repo_root().join("rust/crates/synth_gepa/src/service.rs");
        let raw = std::fs::read_to_string(&path).expect("read service.rs");
        let lexed = Lexed::new(&raw);
        let start = *lexed
            .code_matches("fn public_event_kind(")
            .first()
            .expect("public_event_kind is the projection authority");
        let end = lexed.item_end(start).expect("public_event_kind body");
        let mut found = BTreeSet::new();
        for at in lexed.code_matches("Some(") {
            if at < start || at >= end {
                continue;
            }
            let argument_end = lexed.first_argument_end(at + "Some".len());
            for literal in lexed.strings_within(at + "Some".len() + 1, argument_end) {
                found.insert(literal.to_string());
            }
        }
        found
    }

    fn committed_vocabulary() -> Value {
        let path = repo_root().join("contracts/event_vocabulary.json");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
            panic!(
                "{} is the exported event vocabulary and must be committed: {error}",
                path.display()
            )
        });
        serde_json::from_str(&text).expect("event_vocabulary.json is valid JSON")
    }

    fn as_set(list: &[&str]) -> BTreeSet<String> {
        list.iter().map(|name| name.to_string()).collect()
    }

    #[test]
    fn lexer_ignores_literals_and_comments() {
        let lexed =
            Lexed::new("let x = \"a.emit(\\\"z\\\")\"; // .emit(\"c\")\nfoo.emit(\"real\");");
        let matches = lexed.code_matches(".emit(");
        assert_eq!(matches.len(), 1, "only the real call is code");
        let end = lexed.first_argument_end(matches[0] + ".emit".len());
        assert_eq!(
            lexed.strings_within(matches[0] + ".emit".len() + 1, end),
            vec!["real"]
        );
    }

    #[test]
    fn lexer_distinguishes_char_literals_from_lifetimes() {
        let lexed = Lexed::new("fn f<'a>(c: char) -> bool { c == '}' }");
        assert_eq!(lexed.item_end(0), Some(lexed.source.len()));
    }

    #[test]
    fn declared_constants_are_sorted_and_unique() {
        for list in [OPTIMIZER_EVENT_TYPES, SERVICE_RUN_EVENT_KINDS] {
            let mut sorted = list.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(
                list.to_vec(),
                sorted,
                "vocabulary constants must be sorted and unique"
            );
        }
    }

    #[test]
    fn declared_optimizer_event_types_match_the_emitters() {
        let scanned = scanned_optimizer_event_types();
        let declared = as_set(OPTIMIZER_EVENT_TYPES);
        let undeclared: Vec<&String> = scanned.difference(&declared).collect();
        let unemitted: Vec<&String> = declared.difference(&scanned).collect();
        assert!(
            undeclared.is_empty(),
            "these event types can be emitted but are not in OPTIMIZER_EVENT_TYPES: {undeclared:?}"
        );
        assert!(
            unemitted.is_empty(),
            "these event types are declared but nothing emits them — delete them, \
             do not add an emitter to satisfy this test: {unemitted:?}"
        );
    }

    #[test]
    fn declared_service_run_event_kinds_match_the_projection() {
        assert_eq!(
            scanned_service_run_event_kinds(),
            as_set(SERVICE_RUN_EVENT_KINDS),
            "public_event_kind and SERVICE_RUN_EVENT_KINDS disagree"
        );
    }

    #[test]
    fn committed_contract_matches_the_rust_half() {
        let document = committed_vocabulary();
        assert_eq!(document["schema_version"], "optimizer_event_vocabulary.v1");
        let entries = document["event_types"]
            .as_array()
            .expect("event_types is an array");

        let names: Vec<&str> = entries
            .iter()
            .map(|entry| {
                entry["event_type"]
                    .as_str()
                    .expect("event_type is a string")
            })
            .collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            names, sorted,
            "event_vocabulary.json must be sorted and unique"
        );

        let mut rust_optimizer = BTreeSet::new();
        let mut rust_projection = BTreeSet::new();
        for entry in entries {
            let name = entry["event_type"].as_str().expect("event_type");
            let emitter = entry["emitter"].as_str().expect("emitter");
            assert!(
                emitter == "rust" || emitter == "python",
                "{name}: emitter must be rust or python, got {emitter}"
            );
            let feeds: Vec<&str> = entry["feeds"]
                .as_array()
                .expect("feeds is an array")
                .iter()
                .map(|feed| feed.as_str().expect("feed is a string"))
                .collect();
            assert!(!feeds.is_empty(), "{name}: at least one feed");
            if feeds.contains(&OPTIMIZER_EVENT_FEED) {
                assert_eq!(
                    emitter, "rust",
                    "{name}: {OPTIMIZER_EVENT_FEED} is a Rust feed"
                );
                rust_optimizer.insert(name.to_string());
            }
            if feeds.contains(&SERVICE_RUN_EVENTS_FEED) {
                assert_eq!(
                    emitter, "rust",
                    "{name}: {SERVICE_RUN_EVENTS_FEED} is a Rust feed"
                );
                rust_projection.insert(name.to_string());
            }
        }

        assert_eq!(
            rust_optimizer,
            as_set(OPTIMIZER_EVENT_TYPES),
            "contracts/event_vocabulary.json is stale for {OPTIMIZER_EVENT_FEED}"
        );
        assert_eq!(
            rust_projection,
            as_set(SERVICE_RUN_EVENT_KINDS),
            "contracts/event_vocabulary.json is stale for {SERVICE_RUN_EVENTS_FEED}"
        );
    }
}
