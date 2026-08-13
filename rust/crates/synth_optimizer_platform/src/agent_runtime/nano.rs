use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::{
    usage_from_messages, AgentTurnOutcome, CodexAppServerClient, CodexAppServerLaunch,
    CodexTurnRequest, ExecutionSubstrate, OptimizerError, Result,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NanoAgentTurnIdentity {
    pub request_id: String,
    pub run_id: String,
    pub role: String,
    pub round: String,
    pub treatment_preset: String,
    pub parent_candidate_id: String,
    pub workspace_id: String,
}

pub struct NanoCodexTurnRequest<'a> {
    pub turn: CodexTurnRequest<'a>,
    pub identity: NanoAgentTurnIdentity,
    pub static_context: Value,
    pub replay_artifact_paths: Vec<PathBuf>,
    pub artifact_dir: &'a Path,
    pub cancel_before_start: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NanoCodexEvent {
    pub schema_version: String,
    pub event_id: u64,
    pub timestamp_unix_ms: u128,
    pub session_id: String,
    pub request_id: String,
    pub kind: String,
    pub payload: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NanoCodexLatency {
    pub cold_start_ms: u128,
    pub static_context_load_ms: u128,
    pub first_token_ms: u128,
    pub tool_round_trip_ms: u128,
    pub manifest_validation_ms: u128,
    pub total_turn_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NanoCodexRecordedOutcome {
    pub thread_id: String,
    pub turn_id: String,
    pub thread_response: Value,
    pub final_turn: Value,
    pub usage: Option<Value>,
    pub sent_messages: Vec<Value>,
    pub received_messages: Vec<Value>,
    pub shutdown_warning: Option<String>,
}

impl NanoCodexRecordedOutcome {
    fn from_live(outcome: &AgentTurnOutcome) -> Self {
        Self {
            thread_id: outcome.thread_id.clone(),
            turn_id: outcome.turn_id.clone(),
            thread_response: outcome.thread_response.clone(),
            final_turn: outcome.final_turn.clone(),
            usage: outcome.usage.clone(),
            sent_messages: outcome.sent_messages.clone(),
            received_messages: outcome.received_messages.clone(),
            shutdown_warning: outcome.shutdown_warning.clone(),
        }
    }

    fn into_replay(self) -> AgentTurnOutcome {
        AgentTurnOutcome {
            thread_id: self.thread_id,
            turn_id: self.turn_id,
            thread_response: self.thread_response,
            final_turn: self.final_turn,
            usage: self.usage,
            sent_messages: self.sent_messages,
            received_messages: self.received_messages,
            supervisor_receipt: None,
            shutdown_warning: self.shutdown_warning,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NanoCodexTurnReceipt {
    pub schema_version: String,
    pub mode: String,
    pub identity: NanoAgentTurnIdentity,
    pub session_id: String,
    pub session_reused: bool,
    pub static_context_sha256: String,
    pub static_context_cache_hit: bool,
    pub allowed_tools: Vec<String>,
    pub observed_tools: Vec<String>,
    pub event_id_start: u64,
    pub event_id_end: u64,
    pub terminal_reason: String,
    pub live_model_calls: u64,
    pub live_tool_calls: u64,
    #[serde(default)]
    pub workspace_artifacts: BTreeMap<String, Value>,
    pub latency: NanoCodexLatency,
    pub outcome: NanoCodexRecordedOutcome,
}

pub struct NanoCodexExecution {
    pub outcome: AgentTurnOutcome,
    pub receipt_path: PathBuf,
    pub receipt: NanoCodexTurnReceipt,
}

struct NanoCodexSession {
    client: CodexAppServerClient,
    session_id: String,
    static_context_sha256: String,
    thread_id: String,
    thread_response: Value,
    turn_count: usize,
    next_event_id: u64,
    cold_start_ms: u128,
}

impl Drop for NanoCodexSession {
    fn drop(&mut self) {
        let _ = self.client.persist_refreshed_auth_home();
        let _ = self.client.terminate();
    }
}

#[derive(Default)]
pub struct NanoCodexSessionPool {
    sessions: BTreeMap<String, NanoCodexSession>,
}

impl NanoCodexSessionPool {
    pub fn run(&mut self, request: NanoCodexTurnRequest<'_>) -> Result<NanoCodexExecution> {
        validate_request(&request)?;
        let config = &request.turn.proposer.nano_codex;
        let static_context_sha256 = stable_context_sha256(&request)?;
        let session_key = format!(
            "{}:{}:{}",
            request.identity.run_id, request.identity.role, static_context_sha256
        );
        let record_root = config
            .record_dir
            .clone()
            .unwrap_or_else(|| request.artifact_dir.to_path_buf());
        let turns_dir = record_root.join("turns");
        let sessions_dir = record_root.join("sessions");
        let static_context_dir = record_root.join("static_context");
        fs::create_dir_all(&turns_dir).map_err(|source| OptimizerError::io(&turns_dir, source))?;
        fs::create_dir_all(&sessions_dir)
            .map_err(|source| OptimizerError::io(&sessions_dir, source))?;
        fs::create_dir_all(&static_context_dir)
            .map_err(|source| OptimizerError::io(&static_context_dir, source))?;
        let receipt_path = turns_dir.join(format!(
            "{}.json",
            sanitize_component(&request.identity.request_id)
        ));
        if normalize_mode(&config.mode) == "replay" {
            let replay_root = config.replay_dir.as_ref().ok_or_else(|| {
                OptimizerError::Config(
                    "nano-Codex replay requested without a replay_dir".to_string(),
                )
            })?;
            return replay_turn(&request, replay_root, &receipt_path, &static_context_sha256);
        }

        let static_context_path = static_context_dir.join(format!("{static_context_sha256}.json"));
        let context_started = Instant::now();
        let static_context_cache_hit = static_context_path.is_file();
        if static_context_cache_hit {
            let cached: Value = read_json(&static_context_path)?;
            if cached != request.static_context {
                return Err(OptimizerError::Invariant(format!(
                    "nano-Codex static context digest collision at {}",
                    static_context_path.display()
                )));
            }
        } else {
            write_json(&static_context_path, &request.static_context)?;
        }
        let static_context_load_ms = context_started.elapsed().as_millis();

        let session_expired = self
            .sessions
            .get(&session_key)
            .is_some_and(|session| session.turn_count >= config.max_turns_per_session);
        if session_expired {
            self.sessions.remove(&session_key);
        }
        let session_reused = self.sessions.contains_key(&session_key);
        if !session_reused {
            let session = start_session(&request, &static_context_sha256)?;
            self.sessions.insert(session_key.clone(), session);
        }
        let execution = {
            let session = self.sessions.get_mut(&session_key).ok_or_else(|| {
                OptimizerError::Invariant(
                    "nano-Codex session disappeared after initialization".to_string(),
                )
            })?;
            run_live_turn(
                session,
                &request,
                &receipt_path,
                &sessions_dir,
                static_context_cache_hit,
                static_context_load_ms,
                session_reused,
            )
        };
        if execution.is_err() {
            self.sessions.remove(&session_key);
        }
        execution
    }
}

fn start_session(
    request: &NanoCodexTurnRequest<'_>,
    static_context_sha256: &str,
) -> Result<NanoCodexSession> {
    let started = Instant::now();
    let mut client = CodexAppServerClient::start(CodexAppServerLaunch {
        proposer: request.turn.proposer,
        workspace_dir: request.turn.workspace_dir,
        model: request.turn.model,
    })?;
    let initialize_id = client.send_request(
        "initialize",
        json!({
            "clientInfo": {
                "name": request.turn.client_name,
                "title": request.turn.client_title,
                "version": request.turn.client_version,
            }
        }),
    )?;
    client.wait_for_response(
        initialize_id,
        Duration::from_secs(60),
        request.turn.message_stall_timeout,
    )?;
    client.send_notification("initialized", Value::Null)?;
    let thread_request_id =
        client.send_request("thread/start", request.turn.thread_start_params.clone())?;
    let thread_response = client.wait_for_response(
        thread_request_id,
        Duration::from_secs(60),
        request.turn.message_stall_timeout,
    )?;
    let thread_id = crate::extract_thread_id(&thread_response).ok_or_else(|| {
        OptimizerError::Proposer(format!(
            "nano-Codex thread/start response missing thread id: {thread_response}"
        ))
    })?;
    let session_id = stable_text_sha256(&format!(
        "{}:{}:{}",
        request.identity.run_id, request.identity.role, static_context_sha256
    ));
    Ok(NanoCodexSession {
        client,
        session_id,
        static_context_sha256: static_context_sha256.to_string(),
        thread_id,
        thread_response,
        turn_count: 0,
        next_event_id: 1,
        cold_start_ms: started.elapsed().as_millis(),
    })
}

fn run_live_turn(
    session: &mut NanoCodexSession,
    request: &NanoCodexTurnRequest<'_>,
    receipt_path: &Path,
    sessions_dir: &Path,
    static_context_cache_hit: bool,
    static_context_load_ms: u128,
    session_reused: bool,
) -> Result<NanoCodexExecution> {
    let trace_path = sessions_dir.join(format!("{}.events.jsonl", session.session_id));
    let event_id_start = session.next_event_id;
    append_event(
        session,
        &trace_path,
        request,
        "turn_requested",
        json!({
            "session_reused": session_reused,
            "static_context_sha256": session.static_context_sha256,
            "static_context_cache_hit": static_context_cache_hit,
            "allowed_tools": request.turn.proposer.nano_codex.allowed_tools,
        }),
    )?;
    if request.cancel_before_start {
        append_event(
            session,
            &trace_path,
            request,
            "turn_cancelled",
            json!({"terminal_reason": "cancelled_before_start"}),
        )?;
        return Err(OptimizerError::Proposer(format!(
            "nano-Codex request {} cancelled before start",
            request.identity.request_id
        )));
    }

    let started = Instant::now();
    let sent_start = session.client.sent_messages().len();
    let received_start = session.client.received_messages().len();
    let turn_request_id = session.client.send_request(
        "turn/start",
        turn_params_with_identity(
            request.turn.turn_start_params.clone(),
            &session.thread_id,
            request.turn.workspace_dir,
        )?,
    )?;
    let turn_id = session.client.wait_for_turn_started(
        turn_request_id,
        Duration::from_secs(60),
        request.turn.message_stall_timeout,
    )?;
    let first_token_ms = started.elapsed().as_millis();
    let final_turn = match session.client.wait_for_turn(
        &turn_id,
        request.turn.timeout,
        request.turn.message_stall_timeout,
    ) {
        Ok(value) => value,
        Err(error) => {
            let _ = session.client.interrupt_turn(
                &session.thread_id,
                &turn_id,
                Duration::from_secs(30),
                request.turn.message_stall_timeout,
            );
            append_event(
                session,
                &trace_path,
                request,
                "turn_interrupted",
                json!({"turn_id": turn_id, "error": error.to_string()}),
            )?;
            return Err(error);
        }
    };
    crate::ensure_turn_completed(&final_turn)?;
    let sent_messages = session.client.sent_messages()[sent_start..].to_vec();
    let received_messages = session.client.received_messages()[received_start..].to_vec();
    let usage = usage_from_messages(&received_messages, &turn_id)
        .or_else(|| crate::usage_from_message(&final_turn));
    let observed_tools = observed_tools(&received_messages);
    validate_observed_tools(
        &request.turn.proposer.nano_codex.allowed_tools,
        &observed_tools,
    )?;
    for message in &received_messages {
        append_event(
            session,
            &trace_path,
            request,
            "protocol_received",
            message.clone(),
        )?;
    }
    let total_turn_ms = started.elapsed().as_millis();
    let live_tool_calls = observed_tools.len() as u64;
    let workspace_artifacts =
        capture_workspace_artifacts(request.turn.workspace_dir, &request.replay_artifact_paths)?;
    let tool_round_trip_ms = if live_tool_calls == 0 {
        0
    } else {
        total_turn_ms.saturating_sub(first_token_ms) / u128::from(live_tool_calls)
    };
    append_event(
        session,
        &trace_path,
        request,
        "turn_completed",
        json!({
            "turn_id": turn_id,
            "terminal_reason": "completed",
            "observed_tools": observed_tools,
        }),
    )?;
    session.turn_count = session.turn_count.saturating_add(1);
    let outcome = AgentTurnOutcome {
        thread_id: session.thread_id.clone(),
        turn_id,
        thread_response: session.thread_response.clone(),
        final_turn,
        usage,
        sent_messages,
        received_messages,
        supervisor_receipt: None,
        shutdown_warning: None,
    };
    let receipt = NanoCodexTurnReceipt {
        schema_version: "synth.nano_codex.turn_receipt.v1".to_string(),
        mode: "live".to_string(),
        identity: request.identity.clone(),
        session_id: session.session_id.clone(),
        session_reused,
        static_context_sha256: session.static_context_sha256.clone(),
        static_context_cache_hit,
        allowed_tools: request.turn.proposer.nano_codex.allowed_tools.clone(),
        observed_tools,
        event_id_start,
        event_id_end: session.next_event_id.saturating_sub(1),
        terminal_reason: "completed".to_string(),
        live_model_calls: 1,
        live_tool_calls,
        workspace_artifacts,
        latency: NanoCodexLatency {
            cold_start_ms: if session_reused {
                0
            } else {
                session.cold_start_ms
            },
            static_context_load_ms,
            first_token_ms,
            tool_round_trip_ms,
            manifest_validation_ms: 0,
            total_turn_ms,
        },
        outcome: NanoCodexRecordedOutcome::from_live(&outcome),
    };
    write_json(receipt_path, &receipt)?;
    Ok(NanoCodexExecution {
        outcome,
        receipt_path: receipt_path.to_path_buf(),
        receipt,
    })
}

fn replay_turn(
    request: &NanoCodexTurnRequest<'_>,
    replay_root: &Path,
    output_receipt_path: &Path,
    static_context_sha256: &str,
) -> Result<NanoCodexExecution> {
    let source_path = replay_root.join("turns").join(format!(
        "{}.json",
        sanitize_component(&request.identity.request_id)
    ));
    let source: NanoCodexTurnReceipt = read_json(&source_path)?;
    if source.identity != request.identity {
        return Err(OptimizerError::Invariant(format!(
            "nano-Codex replay identity mismatch for {}",
            source_path.display()
        )));
    }
    if source.static_context_sha256 != static_context_sha256 {
        return Err(OptimizerError::Invariant(format!(
            "nano-Codex replay static context mismatch for {}",
            source_path.display()
        )));
    }
    materialize_workspace_artifacts(request.turn.workspace_dir, &source.workspace_artifacts)?;
    let outcome = source.outcome.clone().into_replay();
    let receipt = NanoCodexTurnReceipt {
        mode: "replay".to_string(),
        live_model_calls: 0,
        live_tool_calls: 0,
        terminal_reason: "replayed".to_string(),
        latency: NanoCodexLatency {
            cold_start_ms: 0,
            static_context_load_ms: 0,
            first_token_ms: 0,
            tool_round_trip_ms: 0,
            manifest_validation_ms: source.latency.manifest_validation_ms,
            total_turn_ms: 0,
        },
        ..source
    };
    write_json(output_receipt_path, &receipt)?;
    Ok(NanoCodexExecution {
        outcome,
        receipt_path: output_receipt_path.to_path_buf(),
        receipt,
    })
}

pub fn record_manifest_validation(
    receipt_path: &Path,
    manifest_validation_ms: u128,
) -> Result<NanoCodexTurnReceipt> {
    let mut receipt: NanoCodexTurnReceipt = read_json(receipt_path)?;
    receipt.latency.manifest_validation_ms = manifest_validation_ms;
    write_json(receipt_path, &receipt)?;
    Ok(receipt)
}

fn validate_request(request: &NanoCodexTurnRequest<'_>) -> Result<()> {
    let proposer = request.turn.proposer;
    if !proposer.nano_codex.enabled {
        return Err(OptimizerError::Config(
            "nano-Codex request requires proposer.nano_codex.enabled = true".to_string(),
        ));
    }
    if !matches!(proposer.runtime_substrate, ExecutionSubstrate::Local) {
        return Err(OptimizerError::Config(
            "nano-Codex supports only the explicit local substrate; fallback is forbidden"
                .to_string(),
        ));
    }
    for (field, value) in [
        ("request_id", request.identity.request_id.as_str()),
        ("run_id", request.identity.run_id.as_str()),
        ("role", request.identity.role.as_str()),
        ("round", request.identity.round.as_str()),
        (
            "treatment_preset",
            request.identity.treatment_preset.as_str(),
        ),
        ("workspace_id", request.identity.workspace_id.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(OptimizerError::Config(format!(
                "nano-Codex identity field {field} must be non-empty"
            )));
        }
    }
    for path in &request.replay_artifact_paths {
        validate_artifact_path(path)?;
    }
    Ok(())
}

fn validate_artifact_path(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(OptimizerError::Config(format!(
            "nano-Codex replay artifact path must be a non-empty relative path without traversal: {}",
            path.display()
        )));
    }
    Ok(())
}

fn capture_workspace_artifacts(
    workspace_dir: &Path,
    paths: &[PathBuf],
) -> Result<BTreeMap<String, Value>> {
    let mut artifacts = BTreeMap::new();
    for path in paths {
        validate_artifact_path(path)?;
        let absolute = workspace_dir.join(path);
        let value = read_json(&absolute)?;
        artifacts.insert(path.to_string_lossy().into_owned(), value);
    }
    Ok(artifacts)
}

fn materialize_workspace_artifacts(
    workspace_dir: &Path,
    artifacts: &BTreeMap<String, Value>,
) -> Result<()> {
    for (path, value) in artifacts {
        let relative = Path::new(path);
        validate_artifact_path(relative)?;
        write_json(&workspace_dir.join(relative), value)?;
    }
    Ok(())
}

fn turn_params_with_identity(
    mut params: Value,
    thread_id: &str,
    workspace_dir: &Path,
) -> Result<Value> {
    let object = params.as_object_mut().ok_or_else(|| {
        OptimizerError::Invariant("nano-Codex turn params must be a JSON object".to_string())
    })?;
    object.insert("threadId".to_string(), Value::String(thread_id.to_string()));
    object.insert(
        "cwd".to_string(),
        Value::String(workspace_dir.display().to_string()),
    );
    Ok(params)
}

fn append_event(
    session: &mut NanoCodexSession,
    trace_path: &Path,
    request: &NanoCodexTurnRequest<'_>,
    kind: &str,
    payload: Value,
) -> Result<()> {
    let event = NanoCodexEvent {
        schema_version: "synth.nano_codex.event.v1".to_string(),
        event_id: session.next_event_id,
        timestamp_unix_ms: now_unix_ms(),
        session_id: session.session_id.clone(),
        request_id: request.identity.request_id.clone(),
        kind: kind.to_string(),
        payload,
    };
    session.next_event_id = session.next_event_id.saturating_add(1);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(trace_path)
        .map_err(|source| OptimizerError::io(trace_path, source))?;
    serde_json::to_writer(&mut file, &event)?;
    file.write_all(b"\n")
        .map_err(|source| OptimizerError::io(trace_path, source))
}

fn observed_tools(messages: &[Value]) -> Vec<String> {
    let mut tools = Vec::new();
    for message in messages {
        if message.get("method").and_then(Value::as_str) == Some("item/started") {
            if let Some(tool) = message
                .pointer("/params/item/type")
                .and_then(Value::as_str)
                .and_then(normalize_tool_name)
            {
                tools.push(tool);
            }
        }
        collect_forbidden_tool_types(message, &mut tools);
    }
    tools
}

fn collect_forbidden_tool_types(value: &Value, tools: &mut Vec<String>) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                if key == "type" {
                    if let Some(tool) = value.as_str().and_then(normalize_forbidden_tool_name) {
                        tools.push(tool);
                    }
                }
                collect_forbidden_tool_types(value, tools);
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_forbidden_tool_types(value, tools);
            }
        }
        _ => {}
    }
}

fn normalize_tool_name(value: &str) -> Option<String> {
    let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
    if normalized.contains("commandexecution")
        || normalized.contains("exec_command")
        || normalized == "exec"
        || normalized.contains("shell")
    {
        Some("exec".to_string())
    } else if normalized.contains("filechange") || normalized.contains("apply_patch") {
        Some("apply_patch".to_string())
    } else if normalized.contains("read_file") || normalized == "read" {
        Some("read".to_string())
    } else if normalized.contains("websearch") || normalized.contains("web_search") {
        Some(format!("unsupported:{normalized}"))
    } else if normalized.contains("search") || normalized.contains("grep") {
        Some("search".to_string())
    } else {
        None
    }
}

fn normalize_forbidden_tool_name(value: &str) -> Option<String> {
    let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
    if normalized.contains("mcp")
        || normalized.contains("toolcall")
        || normalized.contains("tool_call")
        || normalized.contains("websearch")
        || normalized.contains("web_search")
    {
        Some(format!("unsupported:{normalized}"))
    } else {
        None
    }
}

fn validate_observed_tools(allowed: &[String], observed: &[String]) -> Result<()> {
    for tool in observed {
        if !allowed.iter().any(|allowed| allowed == tool) {
            return Err(OptimizerError::Proposer(format!(
                "nano-Codex observed tool {tool:?} outside the configured bounded tool set; no fallback is permitted"
            )));
        }
    }
    Ok(())
}

fn normalize_mode(mode: &str) -> String {
    mode.trim().to_ascii_lowercase().replace('-', "_")
}

fn stable_json_sha256(value: &Value) -> Result<String> {
    let bytes = serde_json::to_vec(value)?;
    Ok(stable_bytes_sha256(&bytes))
}

fn stable_context_sha256(request: &NanoCodexTurnRequest<'_>) -> Result<String> {
    let mut proposer = serde_json::to_value(request.turn.proposer)?;
    let nano = proposer
        .get_mut("nano_codex")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| {
            OptimizerError::Invariant(
                "serialized proposer config is missing nano_codex".to_string(),
            )
        })?;
    // Live and replay must bind the same semantic context. Receipt locations and
    // execution mode affect orchestration only; including them would make a live
    // recording impossible to replay by construction.
    nano.remove("mode");
    nano.remove("record_dir");
    nano.remove("replay_dir");
    stable_json_sha256(&json!({
        "static_context": request.static_context,
        "model": request.turn.model,
        "client_name": request.turn.client_name,
        "client_version": request.turn.client_version,
        "proposer": proposer,
    }))
}

fn stable_text_sha256(value: &str) -> String {
    stable_bytes_sha256(value.as_bytes())
}

fn stable_bytes_sha256(value: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(value);
    format!("{:x}", digest.finalize())
}

fn sanitize_component(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| OptimizerError::io(parent, source))?;
    }
    let text = serde_json::to_string_pretty(value)?;
    fs::write(path, format!("{text}\n")).map_err(|source| OptimizerError::io(path, source))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path).map_err(|source| OptimizerError::io(path, source))?;
    Ok(serde_json::from_str(&text)?)
}
