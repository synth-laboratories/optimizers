use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use crate::{OptimizerError, ProposerConfig, Result};

/// Extra wall-clock budget beyond the per-turn `timeout` for spawn, the three
/// 60s handshake reads, and post-turn auth-persist/terminate. The per-step
/// timeouts only bound the JSON-RPC read loop; this margin lets the hard
/// deadline cover the whole turn lifecycle without firing on a healthy turn.
const TURN_HARD_DEADLINE_MARGIN: Duration = Duration::from_secs(300);

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
    pub workspace_dir: &'a Path,
    pub model: &'a str,
    pub client_name: &'a str,
    pub client_title: &'a str,
    pub client_version: &'a str,
    pub thread_start_params: Value,
    pub turn_start_params: Value,
    pub timeout: Duration,
    pub message_stall_timeout: Duration,
    pub message_observer: Option<AgentMessageObserver>,
}

pub type AgentMessageObserver = Arc<dyn Fn(&Value) -> Result<()> + Send + Sync>;

pub struct CodexCommandExecRequest<'a> {
    pub run_id: &'a str,
    pub proposer: &'a ProposerConfig,
    pub workspace_dir: &'a Path,
    pub model: &'a str,
    pub client_name: &'a str,
    pub client_title: &'a str,
    pub client_version: &'a str,
    pub command: Vec<String>,
    pub command_cwd: Option<String>,
    pub timeout: Duration,
    pub message_stall_timeout: Duration,
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

pub struct AgentCommandExecOutcome {
    pub response: Value,
    pub sent_messages: Vec<Value>,
    pub received_messages: Vec<Value>,
    pub supervisor_receipt: Option<SupervisorReceipt>,
    pub shutdown_warning: Option<String>,
}

impl AgentCommandExecOutcome {
    pub fn exit_code(&self) -> Option<i64> {
        self.response
            .pointer("/result/exitCode")
            .and_then(Value::as_i64)
    }

    pub fn stdout(&self) -> Option<&str> {
        self.response
            .pointer("/result/stdout")
            .and_then(Value::as_str)
    }

    pub fn stderr(&self) -> Option<&str> {
        self.response
            .pointer("/result/stderr")
            .and_then(Value::as_str)
    }
}

pub trait AgentRuntimeSubstrate {
    fn run_codex_turn(&self, request: CodexTurnRequest<'_>) -> Result<AgentTurnOutcome>;

    fn run_codex_command_exec(
        &self,
        request: CodexCommandExecRequest<'_>,
    ) -> Result<AgentCommandExecOutcome> {
        Err(OptimizerError::Proposer(format!(
            "codex app-server command/exec is not implemented for {:?}",
            request.proposer.runtime_substrate
        )))
    }
}

pub fn run_turn(request: CodexTurnRequest<'_>) -> Result<AgentTurnOutcome> {
    match request.proposer.runtime_substrate {
        ExecutionSubstrate::Local => LocalCodexSubstrate.run_codex_turn(request),
        ExecutionSubstrate::Docker => DockerCodexSubstrate.run_codex_turn(request),
        ExecutionSubstrate::Daytona => DaytonaCodexSubstrate.run_codex_turn(request),
    }
}

pub fn run_command_exec(request: CodexCommandExecRequest<'_>) -> Result<AgentCommandExecOutcome> {
    match request.proposer.runtime_substrate {
        ExecutionSubstrate::Local => LocalCodexSubstrate.run_codex_command_exec(request),
        ExecutionSubstrate::Docker => DockerCodexSubstrate.run_codex_command_exec(request),
        ExecutionSubstrate::Daytona => DaytonaCodexSubstrate.run_codex_command_exec(request),
    }
}

pub fn run_codex_jsonrpc_turn(
    mut client: CodexAppServerClient,
    request: CodexTurnRequest<'_>,
    supervisor_receipt: Option<SupervisorReceipt>,
) -> Result<AgentTurnOutcome> {
    // Hard overall deadline. The per-step timeouts bound only the JSON-RPC read
    // loop; a wedge in spawn/auth or post-turn auth-persist/terminate is
    // otherwise unbounded and silently wedges the caller (and, through the
    // parallel-dispatch barrier, the whole run). timeout_means_in_doubt: we do
    // not merely stop waiting — a watchdog SIGKILLs the codex app-server child so
    // every blocked step errors out and the failure bubbles as a typed, in-doubt
    // timeout instead of an infinite hang.
    let child_pid = client.process_id();
    let hard_deadline = request.timeout + TURN_HARD_DEADLINE_MARGIN;
    let done = Arc::new(AtomicBool::new(false));
    let killed = Arc::new(AtomicBool::new(false));
    let watchdog = spawn_turn_hard_deadline_watchdog(
        child_pid,
        request.model.to_string(),
        hard_deadline,
        Arc::clone(&done),
        Arc::clone(&killed),
    );

    let result = run_codex_jsonrpc_turn_inner(&mut client, &request, supervisor_receipt);
    let refresh_result = if result.is_ok() {
        client.persist_refreshed_auth_home().map(|_| ())
    } else {
        Ok(())
    };
    let terminate_result = client.terminate();
    done.store(true, Ordering::SeqCst);
    if let Some(handle) = watchdog {
        let _ = handle.join();
    }

    // If the watchdog killed the child, every inner/terminate error downstream is
    // a symptom of the kill — replace it with one explicit, informative cause.
    if killed.load(Ordering::SeqCst) {
        return Err(OptimizerError::Proposer(format!(
            "codex app-server turn for model {} exceeded the hard deadline of {}s; the \
             app-server child (pid {}) was SIGKILLed to unblock the wedged turn (in-doubt: \
             callee terminated, not just abandoned). The per-turn read timeout did not bound \
             this — the wedge was in spawn/auth or post-turn auth-persist/terminate.",
            request.model,
            hard_deadline.as_secs(),
            child_pid
        )));
    }

    let mut outcome = result?;
    refresh_result?;
    if let Err(error) = terminate_result {
        outcome.shutdown_warning = Some(error.to_string());
    }
    Ok(outcome)
}

pub fn run_codex_jsonrpc_command_exec(
    mut client: CodexAppServerClient,
    request: CodexCommandExecRequest<'_>,
    supervisor_receipt: Option<SupervisorReceipt>,
) -> Result<AgentCommandExecOutcome> {
    let child_pid = client.process_id();
    let hard_deadline = request.timeout + TURN_HARD_DEADLINE_MARGIN;
    let done = Arc::new(AtomicBool::new(false));
    let killed = Arc::new(AtomicBool::new(false));
    let watchdog = spawn_turn_hard_deadline_watchdog(
        child_pid,
        request.model.to_string(),
        hard_deadline,
        Arc::clone(&done),
        Arc::clone(&killed),
    );

    let result = run_codex_jsonrpc_command_exec_inner(&mut client, &request, supervisor_receipt);
    let refresh_result = if result.is_ok() {
        client.persist_refreshed_auth_home().map(|_| ())
    } else {
        Ok(())
    };
    let terminate_result = client.terminate();
    done.store(true, Ordering::SeqCst);
    if let Some(handle) = watchdog {
        let _ = handle.join();
    }

    if killed.load(Ordering::SeqCst) {
        return Err(OptimizerError::Proposer(format!(
            "codex app-server command/exec for model {} exceeded the hard deadline of {}s; the \
             app-server child (pid {}) was SIGKILLed to unblock the wedged command",
            request.model,
            hard_deadline.as_secs(),
            child_pid
        )));
    }

    let mut outcome = result?;
    refresh_result?;
    if let Err(error) = terminate_result {
        outcome.shutdown_warning = Some(error.to_string());
    }
    Ok(outcome)
}

/// Watchdog that SIGKILLs the codex app-server child if the turn lifecycle
/// exceeds `hard_deadline`. Returns early (no kill) once `done` is set, so a
/// healthy turn is never touched. The `done` re-check immediately before the
/// kill shrinks the (already negligible) window where the child has exited and
/// its pid could be reused; a fully reuse-proof kill would hold the `Child`
/// handle, which the turn thread owns exclusively today.
fn spawn_turn_hard_deadline_watchdog(
    child_pid: u32,
    model: String,
    hard_deadline: Duration,
    done: Arc<AtomicBool>,
    killed: Arc<AtomicBool>,
) -> Option<std::thread::JoinHandle<()>> {
    std::thread::Builder::new()
        .name("codex-turn-hard-deadline".to_string())
        .spawn(move || {
            let step = Duration::from_secs(5);
            let mut waited = Duration::ZERO;
            while waited < hard_deadline {
                std::thread::sleep(step);
                if done.load(Ordering::SeqCst) {
                    return;
                }
                waited += step;
            }
            if done.load(Ordering::SeqCst) {
                return;
            }
            killed.store(true, Ordering::SeqCst);
            eprintln!(
                "[codex.turn] HARD DEADLINE model={model} pid={child_pid} exceeded {}s; \
                 sending SIGKILL to unblock the wedged codex turn (in-doubt)",
                hard_deadline.as_secs()
            );
            // SAFETY: SIGKILL to the codex app-server child pid. Guarded by the
            // `done` re-check above so a completed turn's (possibly reused) pid is
            // not signalled in the common path.
            unsafe {
                libc::kill(child_pid as libc::pid_t, libc::SIGKILL);
            }
        })
        .ok()
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
    client.wait_for_response(
        initialize_id,
        Duration::from_secs(60),
        request.message_stall_timeout,
    )?;
    client.send_notification("initialized", Value::Null)?;

    let thread_request_id =
        client.send_request("thread/start", request.thread_start_params.clone())?;
    let thread_response = client.wait_for_response(
        thread_request_id,
        Duration::from_secs(60),
        request.message_stall_timeout,
    )?;
    let thread_id = super::app_server::extract_thread_id(&thread_response).ok_or_else(|| {
        OptimizerError::Proposer(format!(
            "codex app-server thread/start response missing thread id: {thread_response}"
        ))
    })?;

    let turn_request_id = client.send_request(
        "turn/start",
        turn_start_params_with_thread_id(request.turn_start_params.clone(), &thread_id)?,
    )?;
    let turn_id = client.wait_for_turn_started(
        turn_request_id,
        Duration::from_secs(60),
        request.message_stall_timeout,
    )?;
    let final_turn = client.wait_for_turn_with_observer(
        &turn_id,
        request.timeout,
        request.message_stall_timeout,
        request.message_observer.as_ref(),
    )?;
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

fn run_codex_jsonrpc_command_exec_inner(
    client: &mut CodexAppServerClient,
    request: &CodexCommandExecRequest<'_>,
    supervisor_receipt: Option<SupervisorReceipt>,
) -> Result<AgentCommandExecOutcome> {
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
    client.wait_for_response(
        initialize_id,
        Duration::from_secs(60),
        request.message_stall_timeout,
    )?;
    client.send_notification("initialized", Value::Null)?;

    let command_cwd = request
        .command_cwd
        .clone()
        .unwrap_or_else(|| request.workspace_dir.display().to_string());
    let command_request_id =
        client.send_request("command/exec", command_exec_params(request, command_cwd))?;
    let response = client.wait_for_response(
        command_request_id,
        request.timeout,
        request.message_stall_timeout,
    )?;

    Ok(AgentCommandExecOutcome {
        response,
        sent_messages: client.sent_messages().to_vec(),
        received_messages: client.received_messages().to_vec(),
        supervisor_receipt,
        shutdown_warning: None,
    })
}

pub(crate) fn command_exec_params(
    request: &CodexCommandExecRequest<'_>,
    command_cwd: String,
) -> Value {
    let timeout_ms = request.timeout.as_millis().max(1).min(i64::MAX as u128) as i64;
    let mut params = json!({
        "command": request.command.clone(),
        "cwd": command_cwd,
        "timeoutMs": timeout_ms,
    });
    if let Some(sandbox_mode) = request
        .proposer
        .sandbox_mode
        .as_deref()
        .map(str::trim)
        .filter(|mode| !mode.is_empty())
    {
        params["sandboxPolicy"] = super::role_agent::sandbox_policy_for_mode(sandbox_mode);
    }
    params
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
