use std::time::Duration;

use serde_json::{json, Value};

use crate::{OptimizerError, ProposerConfig, Result};

use super::app_server::CodexAppServerClient;
use super::daytona::DaytonaCodexSubstrate;
use super::docker::DockerCodexSubstrate;
use super::local::LocalCodexSubstrate;
use super::substrate::ExecutionSubstrate;
use super::supervisor::SupervisorReceipt;
use super::usage::{usage_from_message, usage_from_messages};

pub struct CodexTurnRequest<'a> {
    pub run_id: &'a str,
    pub proposer: &'a ProposerConfig,
    pub workspace_dir: &'a std::path::Path,
    pub model: &'a str,
    pub client_name: &'a str,
    pub client_title: &'a str,
    pub client_version: &'a str,
    pub thread_start_params: Value,
    pub turn_start_params: Value,
    pub timeout: Duration,
}

pub struct AgentTurnOutcome {
    pub thread_id: String,
    pub turn_id: String,
    pub thread_response: Value,
    pub final_turn: Value,
    pub usage: Option<Value>,
    pub sent_messages: Vec<Value>,
    pub received_messages: Vec<Value>,
    pub supervisor_receipt: Option<SupervisorReceipt>,
    pub shutdown_warning: Option<String>,
}

pub trait AgentRuntimeSubstrate {
    fn run_codex_turn(&self, request: CodexTurnRequest<'_>) -> Result<AgentTurnOutcome>;
}

pub fn run_turn(request: CodexTurnRequest<'_>) -> Result<AgentTurnOutcome> {
    match request.proposer.runtime_substrate {
        ExecutionSubstrate::Local => LocalCodexSubstrate.run_codex_turn(request),
        ExecutionSubstrate::Docker => DockerCodexSubstrate.run_codex_turn(request),
        ExecutionSubstrate::Daytona => DaytonaCodexSubstrate.run_codex_turn(request),
    }
}

pub fn run_codex_jsonrpc_turn(
    mut client: CodexAppServerClient,
    request: CodexTurnRequest<'_>,
    supervisor_receipt: Option<SupervisorReceipt>,
) -> Result<AgentTurnOutcome> {
    let result = run_codex_jsonrpc_turn_inner(&mut client, &request, supervisor_receipt);
    let terminate_result = client.terminate();
    let mut outcome = result?;
    if let Err(error) = terminate_result {
        outcome.shutdown_warning = Some(error.to_string());
    }
    Ok(outcome)
}

fn run_codex_jsonrpc_turn_inner(
    client: &mut CodexAppServerClient,
    request: &CodexTurnRequest<'_>,
    supervisor_receipt: Option<SupervisorReceipt>,
) -> Result<AgentTurnOutcome> {
    let initialize_id = client.send_request(
        "initialize",
        json!({
            "clientInfo": {
                "name": request.client_name,
                "title": request.client_title,
                "version": request.client_version,
            }
        }),
    )?;
    client.wait_for_response(initialize_id, Duration::from_secs(60))?;
    client.send_notification("initialized", Value::Null)?;

    let thread_request_id =
        client.send_request("thread/start", request.thread_start_params.clone())?;
    let thread_response = client.wait_for_response(thread_request_id, Duration::from_secs(60))?;
    let thread_id = super::app_server::extract_thread_id(&thread_response).ok_or_else(|| {
        OptimizerError::Proposer(format!(
            "codex app-server thread/start response missing thread id: {thread_response}"
        ))
    })?;

    let turn_request_id = client.send_request(
        "turn/start",
        turn_start_params_with_thread_id(request.turn_start_params.clone(), &thread_id)?,
    )?;
    let turn_id = client.wait_for_turn_started(turn_request_id, Duration::from_secs(60))?;
    let final_turn = client.wait_for_turn(&turn_id, request.timeout)?;
    super::app_server::ensure_turn_completed(&final_turn)?;
    let usage = usage_from_messages(client.received_messages(), &turn_id)
        .or_else(|| usage_from_message(&final_turn));

    Ok(AgentTurnOutcome {
        thread_id,
        turn_id,
        thread_response,
        final_turn,
        usage,
        sent_messages: client.sent_messages().to_vec(),
        received_messages: client.received_messages().to_vec(),
        supervisor_receipt,
        shutdown_warning: None,
    })
}

fn turn_start_params_with_thread_id(mut params: Value, thread_id: &str) -> Result<Value> {
    let Some(map) = params.as_object_mut() else {
        return Err(OptimizerError::Proposer(format!(
            "codex turn/start params must be an object; got {params}"
        )));
    };
    if map.contains_key("threadId") {
        return Err(OptimizerError::Proposer(
            "codex turn/start params must not set threadId before platform runtime binding"
                .to_string(),
        ));
    }
    map.insert("threadId".to_string(), Value::String(thread_id.to_string()));
    Ok(params)
}
