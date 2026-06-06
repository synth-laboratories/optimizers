use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use reqwest::blocking::multipart;
use serde_json::{json, Value};

use crate::{OptimizerError, ProposerConfig, ProposerDaytonaConfig, Result};

use super::app_server::{ensure_turn_completed, extract_thread_id};
use super::codex_home::prepare_proposer_codex_launch;
use super::limits;
use super::session::{AgentRuntimeSubstrate, AgentTurnOutcome, CodexTurnRequest};
use super::supervisor::SupervisorReceipt;
use super::usage::{usage_from_message, usage_from_messages};

const STDOUT_PREFIX: &[u8] = b"\x01\x01\x01";
const STDERR_PREFIX: &[u8] = b"\x02\x02\x02";
const UPLOAD_BATCH_SIZE: usize = 64;

pub struct DaytonaCodexSubstrate;

impl AgentRuntimeSubstrate for DaytonaCodexSubstrate {
    fn run_codex_turn(&self, request: CodexTurnRequest<'_>) -> Result<AgentTurnOutcome> {
        run_daytona_codex_turn(request)
    }
}

fn run_daytona_codex_turn(request: CodexTurnRequest<'_>) -> Result<AgentTurnOutcome> {
    let daytona = request.proposer.daytona.as_ref().ok_or_else(|| {
        OptimizerError::Config(
            "proposer.runtime_substrate = \"daytona\" requires [proposer.daytona]".to_string(),
        )
    })?;
    let original_workspace = fs::canonicalize(request.workspace_dir)
        .map_err(|source| OptimizerError::io(request.workspace_dir, source))?;
    let staged_workspace = stage_workspace(&original_workspace, request.run_id)?;
    let mut receipt = SupervisorReceipt {
        substrate: "daytona".to_string(),
        process_id: None,
        container_name: None,
        image: daytona.image.clone(),
        staging_dir: Some(staged_workspace.display().to_string()),
        workspace_mount_path: Some(daytona.remote_workspace_dir.clone()),
        cleanup_status: "pending".to_string(),
        sandbox_id: None,
        sandbox_name: None,
        daytona_target: daytona.target.clone(),
        command_id: None,
        toolbox_url: None,
    };
    let run_result =
        run_daytona_with_staged_workspace(request, daytona, &staged_workspace, &mut receipt);
    let sync_result = if daytona.sync_workspace_back {
        sync_workspace_back(&staged_workspace, &original_workspace)
    } else {
        Ok(())
    };
    let cleanup_result = cleanup_staged_workspace(&staged_workspace);
    match (run_result, sync_result, cleanup_result) {
        (Ok(mut outcome), Ok(()), Ok(())) => {
            if let Some(receipt) = outcome.supervisor_receipt.as_mut() {
                receipt.cleanup_status = if daytona.keep_sandbox {
                    "sandbox_kept".to_string()
                } else {
                    "cleaned".to_string()
                };
            }
            Ok(outcome)
        }
        (Err(error), Ok(()), Ok(())) => Err(error),
        (result, sync, cleanup) => Err(OptimizerError::Proposer(format!(
            "daytona proposer workspace cleanup failed: run_result={}; sync_result={}; cleanup_result={}",
            result_status(&result),
            result_status(&sync),
            result_status(&cleanup)
        ))),
    }
}

fn run_daytona_with_staged_workspace(
    request: CodexTurnRequest<'_>,
    daytona: &ProposerDaytonaConfig,
    staged_workspace: &Path,
    receipt: &mut SupervisorReceipt,
) -> Result<AgentTurnOutcome> {
    let api_key = env::var(&daytona.api_key_env)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            OptimizerError::Config(format!(
                "proposer.runtime_substrate = \"daytona\" requires non-empty {}",
                daytona.api_key_env
            ))
        })?;
    let mut host_env = env::vars().collect::<BTreeMap<_, _>>();
    let launch_state = prepare_proposer_codex_launch(
        request.proposer,
        staged_workspace,
        request.model,
        host_env.clone(),
    )?;
    host_env.extend(launch_state.env_map.clone());
    let daytona_client = DaytonaControlClient::new(daytona, api_key)?;
    let sandbox = daytona_client.create_sandbox(request.run_id, &sandbox_env(daytona)?)?;
    receipt.sandbox_id = Some(sandbox.id.clone());
    receipt.sandbox_name = sandbox.name.clone();
    receipt.daytona_target = sandbox.target.clone().or_else(|| daytona.target.clone());
    daytona_client.wait_for_started(&sandbox.id)?;
    let toolbox_url = daytona_client.toolbox_url(&sandbox.id)?;
    receipt.toolbox_url = Some(toolbox_url.clone());
    let toolbox = DaytonaToolboxClient::new(
        toolbox_url,
        sandbox.id.clone(),
        daytona_client.auth_headers(),
    );
    toolbox.exec_shell(&format!(
        "rm -rf {workspace} && mkdir -p {workspace}",
        workspace = shell_quote(&daytona.remote_workspace_dir)
    ))?;
    toolbox.upload_workspace(staged_workspace, &daytona.remote_workspace_dir)?;
    let session_id = format!("codex-app-server-{}", safe_fragment(request.run_id));
    toolbox.create_session(&session_id)?;
    let remote_env = remote_codex_env(daytona, &launch_state, &host_env)?;
    let command = bootstrap_command(daytona, request.proposer, &remote_env);
    let command_id = toolbox.execute_session_command(&session_id, &command, true)?;
    receipt.command_id = Some(command_id.clone());
    eprintln!(
        "[gepa-proposer] daytona substrate started run_id={} sandbox={} command={} workspace={}",
        request.run_id, sandbox.id, command_id, daytona.remote_workspace_dir
    );

    let mut client = DaytonaAppServerClient::new(
        toolbox,
        session_id,
        command_id,
        Duration::from_millis(daytona.poll_interval_ms),
    );
    let result = run_daytona_jsonrpc_turn(&mut client, request, Some(receipt.clone()));
    let terminate_result = if daytona.keep_sandbox {
        Ok(())
    } else {
        client.terminate()
    };
    let delete_result = if daytona.keep_sandbox {
        Ok(())
    } else {
        daytona_client.delete_sandbox(&sandbox.id)
    };
    let mut outcome = result?;
    if let Err(error) = terminate_result {
        outcome.shutdown_warning = Some(error.to_string());
    }
    if let Err(error) = delete_result {
        outcome.shutdown_warning = Some(match outcome.shutdown_warning.take() {
            Some(existing) => format!("{existing}; daytona sandbox delete failed: {error}"),
            None => format!("daytona sandbox delete failed: {error}"),
        });
        if let Some(receipt) = outcome.supervisor_receipt.as_mut() {
            receipt.cleanup_status = "sandbox_delete_failed".to_string();
        }
    }
    Ok(outcome)
}

fn sandbox_env(daytona: &ProposerDaytonaConfig) -> Result<BTreeMap<String, String>> {
    let mut env_map = daytona.env.clone();
    for (sandbox_key, host_key) in &daytona.extra_env {
        let value = env::var(host_key).map_err(|source| {
            OptimizerError::Config(format!(
                "proposer.daytona.extra_env maps {sandbox_key} to host env {host_key}, but it is unavailable: {source}"
            ))
        })?;
        if value.trim().is_empty() {
            return Err(OptimizerError::Config(format!(
                "proposer.daytona.extra_env maps {sandbox_key} to host env {host_key}, but it is empty"
            )));
        }
        env_map.insert(sandbox_key.clone(), value);
    }
    Ok(env_map)
}

fn remote_codex_env(
    daytona: &ProposerDaytonaConfig,
    launch_state: &super::codex_home::ProposerCodexLaunch,
    host_env: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>> {
    let mut env_map = BTreeMap::new();
    env_map.insert(
        "SYNTH_WORKSPACE".to_string(),
        daytona.remote_workspace_dir.clone(),
    );
    let codex_home = launch_state
        .codex_home_workspace_relative_path
        .as_ref()
        .ok_or_else(|| {
            OptimizerError::Proposer(
                "daytona proposer auth preparation did not produce a workspace-relative CODEX_HOME"
                    .to_string(),
            )
        })?;
    env_map.insert(
        "CODEX_HOME".to_string(),
        remote_join(&daytona.remote_workspace_dir, codex_home),
    );
    if let Some(api_key) = host_env.get("OPENAI_API_KEY") {
        env_map.insert("OPENAI_API_KEY".to_string(), api_key.clone());
    }
    Ok(env_map)
}

fn bootstrap_command(
    daytona: &ProposerDaytonaConfig,
    proposer: &ProposerConfig,
    env_map: &BTreeMap<String, String>,
) -> String {
    let mut parts = vec!["env".to_string()];
    for (key, value) in env_map {
        parts.push(format!("{key}={value}"));
    }
    parts.push("bash".to_string());
    parts.push("-c".to_string());
    parts.push(format!(
        "cd {} && exec {}",
        shell_quote(&daytona.remote_workspace_dir),
        shell_join(&inner_codex_command(proposer))
    ));
    shell_join(&parts)
}

fn run_daytona_jsonrpc_turn(
    client: &mut DaytonaAppServerClient,
    request: CodexTurnRequest<'_>,
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
    let thread_id = extract_thread_id(&thread_response).ok_or_else(|| {
        OptimizerError::Proposer(format!(
            "daytona codex app-server thread/start response missing thread id: {thread_response}"
        ))
    })?;

    let turn_request_id = client.send_request(
        "turn/start",
        turn_start_params_with_thread_id(request.turn_start_params.clone(), &thread_id)?,
    )?;
    let turn_id = client.wait_for_turn_started(turn_request_id, Duration::from_secs(60))?;
    let final_turn = client.wait_for_turn(&turn_id, request.timeout)?;
    ensure_turn_completed(&final_turn)?;
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
    if !params.is_object() {
        return Err(OptimizerError::Proposer(format!(
            "turn/start params must be an object: {params}"
        )));
    }
    let object = params.as_object_mut().expect("checked object above");
    object.insert("threadId".to_string(), Value::String(thread_id.to_string()));
    Ok(params)
}

struct DaytonaAppServerClient {
    toolbox: DaytonaToolboxClient,
    session_id: String,
    command_id: String,
    poll_interval: Duration,
    next_id: u64,
    sent_messages: Vec<Value>,
    received_messages: Vec<Value>,
    buffer: VecDeque<Value>,
    stdout_offset: usize,
    stderr_offset: usize,
    stdout_remainder: String,
    stderr_tail: VecDeque<String>,
}

impl DaytonaAppServerClient {
    fn new(
        toolbox: DaytonaToolboxClient,
        session_id: String,
        command_id: String,
        poll_interval: Duration,
    ) -> Self {
        Self {
            toolbox,
            session_id,
            command_id,
            poll_interval,
            next_id: 1,
            sent_messages: Vec::new(),
            received_messages: Vec::new(),
            buffer: VecDeque::new(),
            stdout_offset: 0,
            stderr_offset: 0,
            stdout_remainder: String::new(),
            stderr_tail: VecDeque::new(),
        }
    }

    fn sent_messages(&self) -> &[Value] {
        &self.sent_messages
    }

    fn received_messages(&self) -> &[Value] {
        &self.received_messages
    }

    fn send_request(&mut self, method: &str, params: Value) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))?;
        Ok(id)
    }

    fn send_notification(&mut self, method: &str, params: Value) -> Result<()> {
        self.send(json!({"jsonrpc": "2.0", "method": method, "params": params}))
    }

    fn send(&mut self, message: Value) -> Result<()> {
        let payload = serde_json::to_string(&message)?;
        self.toolbox
            .send_session_input(&self.session_id, &self.command_id, &(payload + "\n"))?;
        self.sent_messages.push(message);
        Ok(())
    }

    fn wait_for_response(&mut self, id: u64, timeout: Duration) -> Result<Value> {
        let deadline = Instant::now() + timeout;
        let mut deferred = Vec::new();
        loop {
            let message = self.read_next(deadline)?;
            if message.get("id").and_then(Value::as_u64) == Some(id)
                && message.get("method").is_none()
            {
                if let Some(error) = message.get("error") {
                    return Err(OptimizerError::Proposer(format!(
                        "daytona codex app-server request {id} failed: {error}"
                    )));
                }
                self.restore_deferred(deferred);
                return Ok(message);
            }
            deferred.push(message);
        }
    }

    fn wait_for_turn_started(&mut self, request_id: u64, timeout: Duration) -> Result<String> {
        let deadline = Instant::now() + timeout;
        let mut deferred = Vec::new();
        loop {
            let message = self.read_next(deadline)?;
            if message.get("id").and_then(Value::as_u64) == Some(request_id)
                && message.get("method").is_none()
            {
                if let Some(error) = message.get("error") {
                    return Err(OptimizerError::Proposer(format!(
                        "daytona codex app-server turn/start request failed: {error}"
                    )));
                }
                let turn_id = extract_turn_id(&message).ok_or_else(|| {
                    OptimizerError::Proposer(format!(
                        "daytona codex app-server turn/start response missing turn id: {message}"
                    ))
                })?;
                self.restore_deferred(deferred);
                return Ok(turn_id);
            }
            if message.get("method").and_then(Value::as_str) == Some("turn/started") {
                if let Some(turn_id) = extract_turn_id(&message) {
                    self.restore_deferred(deferred);
                    return Ok(turn_id);
                }
            }
            deferred.push(message);
        }
    }

    fn wait_for_turn(&mut self, turn_id: &str, timeout: Duration) -> Result<Value> {
        let deadline = Instant::now() + timeout;
        loop {
            let message = self.read_next(deadline)?;
            if message_matches_turn(&message, turn_id) {
                if is_terminal_turn_event(&message) {
                    return Ok(message);
                }
            }
        }
    }

    fn terminate(&mut self) -> Result<()> {
        self.toolbox.delete_session(&self.session_id)
    }

    fn read_next(&mut self, deadline: Instant) -> Result<Value> {
        if let Some(message) = self.buffer.pop_front() {
            return Ok(message);
        }
        loop {
            self.poll_logs()?;
            if let Some(message) = self.buffer.pop_front() {
                return Ok(message);
            }
            if let Some(exit_code) = self
                .toolbox
                .command_exit_code(&self.session_id, &self.command_id)?
            {
                return Err(OptimizerError::Proposer(format!(
                    "daytona codex app-server exited before next JSON-RPC message; exit_code={exit_code}; stderr_tail={}",
                    self.stderr_tail.iter().cloned().collect::<Vec<_>>().join("\\n")
                )));
            }
            if Instant::now() >= deadline {
                return Err(OptimizerError::Proposer(format!(
                    "timed out waiting for daytona codex app-server message; stderr_tail={}",
                    self.stderr_tail
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("\\n")
                )));
            }
            thread::sleep(self.poll_interval);
        }
    }

    fn restore_deferred(&mut self, deferred: Vec<Value>) {
        let mut restored = VecDeque::from(deferred);
        restored.append(&mut self.buffer);
        self.buffer = restored;
    }

    fn poll_logs(&mut self) -> Result<()> {
        let logs = self
            .toolbox
            .session_command_logs(&self.session_id, &self.command_id)?;
        let (stdout, stderr) = demux_log(&logs);
        if stdout.len() > self.stdout_offset {
            let chunk = String::from_utf8_lossy(&stdout[self.stdout_offset..]).to_string();
            self.stdout_offset = stdout.len();
            self.handle_stdout_chunk(&chunk)?;
        }
        if stderr.len() > self.stderr_offset {
            let chunk = String::from_utf8_lossy(&stderr[self.stderr_offset..]).to_string();
            self.stderr_offset = stderr.len();
            self.handle_stderr_chunk(&chunk);
        }
        Ok(())
    }

    fn handle_stdout_chunk(&mut self, chunk: &str) -> Result<()> {
        self.stdout_remainder.push_str(chunk);
        loop {
            let trimmed_start = self
                .stdout_remainder
                .trim_start_matches(|c| c == '\r' || c == '\n');
            if trimmed_start.len() != self.stdout_remainder.len() {
                self.stdout_remainder = trimmed_start.to_string();
            }
            if self.stdout_remainder.is_empty() {
                return Ok(());
            }
            if self.stdout_remainder.starts_with('{') || self.stdout_remainder.starts_with('[') {
                let Some(index) = self.stdout_remainder.find('\n') else {
                    return Ok(());
                };
                let line = self.stdout_remainder[..index]
                    .trim_end_matches('\r')
                    .to_string();
                self.stdout_remainder = self.stdout_remainder[index + 1..].to_string();
                self.emit_stdout_line(&line)?;
                continue;
            }
            let Some((header_end, separator_len)) =
                content_length_header_end(&self.stdout_remainder)
            else {
                if let Some(index) = self.stdout_remainder.find('\n') {
                    let line = self.stdout_remainder[..index]
                        .trim_end_matches('\r')
                        .to_string();
                    self.stdout_remainder = self.stdout_remainder[index + 1..].to_string();
                    self.emit_stdout_line(&line)?;
                    continue;
                }
                return Ok(());
            };
            let headers = &self.stdout_remainder[..header_end];
            let Some(content_length) = parse_content_length_header(headers)? else {
                let consumed = header_end + separator_len;
                let line = self.stdout_remainder[..header_end].to_string();
                self.stdout_remainder = self.stdout_remainder[consumed..].to_string();
                self.emit_stdout_line(&line)?;
                continue;
            };
            let body_start = header_end + separator_len;
            let body_end = body_start + content_length;
            if self.stdout_remainder.as_bytes().len() < body_end {
                return Ok(());
            }
            let payload = self.stdout_remainder.as_bytes()[body_start..body_end].to_vec();
            let value: Value = serde_json::from_slice(&payload)?;
            self.received_messages.push(value.clone());
            self.buffer.push_back(value);
            self.stdout_remainder = self.stdout_remainder[body_end..].to_string();
        }
    }

    fn handle_stderr_chunk(&mut self, chunk: &str) {
        for line in chunk.lines() {
            let line = line.trim_end_matches('\r').trim();
            if line.is_empty() {
                continue;
            }
            self.stderr_tail.push_back(line.to_string());
            while self.stderr_tail.len() > 50 {
                self.stderr_tail.pop_front();
            }
        }
    }

    fn emit_stdout_line(&mut self, line: &str) -> Result<()> {
        let normalized = line.trim();
        if normalized.is_empty() {
            return Ok(());
        }
        match serde_json::from_str::<Value>(normalized) {
            Ok(Value::Object(_)) => {
                let value: Value = serde_json::from_str(normalized)?;
                self.received_messages.push(value.clone());
                self.buffer.push_back(value);
            }
            _ => {
                self.stderr_tail.push_back(line.to_string());
                while self.stderr_tail.len() > 50 {
                    self.stderr_tail.pop_front();
                }
            }
        }
        Ok(())
    }
}

fn content_length_header_end(buffer: &str) -> Option<(usize, usize)> {
    if let Some(index) = buffer.find("\r\n\r\n") {
        return Some((index, 4));
    }
    buffer.find("\n\n").map(|index| (index, 2))
}

fn parse_content_length_header(headers: &str) -> Result<Option<usize>> {
    for line in headers.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if key.trim().eq_ignore_ascii_case("content-length") {
            let value = value.trim();
            let len = value.parse::<usize>().map_err(|source| {
                OptimizerError::Proposer(format!(
                    "invalid daytona codex app-server Content-Length {value}: {source}"
                ))
            })?;
            return Ok(Some(len));
        }
    }
    Ok(None)
}

#[derive(Clone)]
struct DaytonaControlClient {
    client: reqwest::blocking::Client,
    api_url: String,
    api_key: String,
    target: Option<String>,
    config: ProposerDaytonaConfig,
}

impl DaytonaControlClient {
    fn new(config: &ProposerDaytonaConfig, api_key: String) -> Result<Self> {
        Ok(Self {
            client: reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(config.startup_timeout_seconds.max(1)))
                .build()?,
            api_url: config.api_url.trim_end_matches('/').to_string(),
            api_key,
            target: config.target.clone(),
            config: config.clone(),
        })
    }

    fn auth_headers(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            (
                "Authorization".to_string(),
                format!("Bearer {}", self.api_key),
            ),
            (
                "X-Daytona-Source".to_string(),
                "synth-optimizers-rust".to_string(),
            ),
        ])
    }

    fn create_sandbox(
        &self,
        run_id: &str,
        env_map: &BTreeMap<String, String>,
    ) -> Result<DaytonaSandbox> {
        let name = format!(
            "{}-{}-{}",
            safe_fragment(&self.config.sandbox_name_prefix),
            safe_fragment(run_id),
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        );
        let mut body = json!({
            "name": name,
            "env": env_map,
            "labels": {
                "synth.run_id": run_id,
                "synth.runtime": "optimizer-proposer",
                "synth.substrate": "daytona"
            },
            "public": self.config.public,
            "target": self.target,
            "autoStopInterval": self.config.auto_stop_interval_minutes,
        });
        if let Some(dockerfile_content) = self
            .config
            .dockerfile_content
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            body["buildInfo"] = json!({
                "dockerfileContent": dockerfile_content,
            });
        } else if let Some(image) = self
            .config
            .image
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            body["buildInfo"] = json!({
                "dockerfileContent": format!("FROM {image}\n"),
            });
        } else if let Some(snapshot) = self
            .config
            .snapshot
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            body["snapshot"] = Value::String(snapshot.to_string());
        }
        let value = self.api_post("/sandbox", &body)?;
        DaytonaSandbox::from_value(value)
    }

    fn wait_for_started(&self, sandbox_id: &str) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(self.config.startup_timeout_seconds);
        loop {
            let sandbox = DaytonaSandbox::from_value(
                self.api_get(&format!("/sandbox/{}", url_fragment(sandbox_id)))?,
            )?;
            match sandbox.state.as_deref() {
                Some("started") => return Ok(()),
                Some("error") | Some("build_failed") => {
                    return Err(OptimizerError::Proposer(format!(
                        "daytona sandbox {sandbox_id} failed to start: state={:?} error_reason={:?}",
                        sandbox.state, sandbox.error_reason
                    )))
                }
                _ => {}
            }
            if Instant::now() >= deadline {
                return Err(OptimizerError::Proposer(format!(
                    "daytona sandbox {sandbox_id} did not reach started before timeout"
                )));
            }
            thread::sleep(Duration::from_millis(self.config.poll_interval_ms));
        }
    }

    fn toolbox_url(&self, sandbox_id: &str) -> Result<String> {
        let value = self.api_get(&format!(
            "/sandbox/{}/toolbox-proxy-url",
            url_fragment(sandbox_id)
        ))?;
        value
            .get("url")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .ok_or_else(|| {
                OptimizerError::Proposer(format!(
                    "daytona sandbox {sandbox_id} toolbox proxy response missing url: {value}"
                ))
            })
    }

    fn delete_sandbox(&self, sandbox_id: &str) -> Result<()> {
        let url = format!("{}/sandbox/{}", self.api_url, url_fragment(sandbox_id));
        let response = self
            .client
            .delete(url)
            .headers(header_map(&self.auth_headers())?)
            .send()?;
        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            Err(OptimizerError::Proposer(format!(
                "daytona sandbox delete failed status={status}: {body}"
            )))
        }
    }

    fn api_get(&self, path: &str) -> Result<Value> {
        let url = format!("{}{}", self.api_url, path);
        let response = self
            .client
            .get(url)
            .headers(header_map(&self.auth_headers())?)
            .send()?;
        json_response(path, response)
    }

    fn api_post(&self, path: &str, body: &Value) -> Result<Value> {
        let url = format!("{}{}", self.api_url, path);
        let response = self
            .client
            .post(url)
            .headers(header_map(&self.auth_headers())?)
            .json(body)
            .send()?;
        json_response(path, response)
    }
}

#[derive(Clone)]
struct DaytonaToolboxClient {
    client: reqwest::blocking::Client,
    toolbox_url: String,
    sandbox_id: String,
    headers: BTreeMap<String, String>,
}

impl DaytonaToolboxClient {
    fn new(toolbox_url: String, sandbox_id: String, headers: BTreeMap<String, String>) -> Self {
        Self {
            client: reqwest::blocking::Client::new(),
            toolbox_url: toolbox_url.trim_end_matches('/').to_string(),
            sandbox_id,
            headers,
        }
    }

    fn exec_shell(&self, command: &str) -> Result<Value> {
        self.post_json(
            "/process/execute",
            &json!({
                "command": format!("sh -lc {}", shell_quote(command)),
                "timeout": 600,
            }),
        )
    }

    fn create_session(&self, session_id: &str) -> Result<()> {
        self.post_json("/process/session", &json!({"sessionId": session_id}))?;
        Ok(())
    }

    fn execute_session_command(
        &self,
        session_id: &str,
        command: &str,
        run_async: bool,
    ) -> Result<String> {
        let value = self.post_json(
            &format!("/process/session/{}/exec", url_fragment(session_id)),
            &json!({
                "command": command,
                "runAsync": run_async,
            }),
        )?;
        value
            .get("cmdId")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .ok_or_else(|| {
                OptimizerError::Proposer(format!(
                    "daytona session execute response missing cmdId: {value}"
                ))
            })
    }

    fn send_session_input(&self, session_id: &str, command_id: &str, data: &str) -> Result<()> {
        self.post_json(
            &format!(
                "/process/session/{}/command/{}/input",
                url_fragment(session_id),
                url_fragment(command_id)
            ),
            &json!({"data": data}),
        )?;
        Ok(())
    }

    fn session_command_logs(&self, session_id: &str, command_id: &str) -> Result<Vec<u8>> {
        let path = format!(
            "/process/session/{}/command/{}/logs?follow=false",
            url_fragment(session_id),
            url_fragment(command_id)
        );
        let url = self.toolbox_path(&path);
        let response = self
            .client
            .get(url)
            .headers(header_map(&self.headers)?)
            .send()?;
        if response.status().is_success() {
            Ok(response.bytes()?.to_vec())
        } else {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            Err(OptimizerError::Proposer(format!(
                "daytona toolbox GET {path} failed status={status}: {body}"
            )))
        }
    }

    fn command_exit_code(&self, session_id: &str, command_id: &str) -> Result<Option<i64>> {
        let value = self.get_json(&format!(
            "/process/session/{}/command/{}",
            url_fragment(session_id),
            url_fragment(command_id)
        ))?;
        Ok(value.get("exitCode").and_then(Value::as_i64))
    }

    fn delete_session(&self, session_id: &str) -> Result<()> {
        let path = format!("/process/session/{}", url_fragment(session_id));
        let url = self.toolbox_path(&path);
        let response = self
            .client
            .delete(url)
            .headers(header_map(&self.headers)?)
            .send()?;
        if response.status().is_success() || response.status().as_u16() == 404 {
            Ok(())
        } else {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            Err(OptimizerError::Proposer(format!(
                "daytona toolbox DELETE {path} failed status={status}: {body}"
            )))
        }
    }

    fn upload_workspace(&self, local_workspace: &Path, remote_workspace: &str) -> Result<()> {
        let files = collect_workspace_files(local_workspace)?;
        for chunk in files.chunks(UPLOAD_BATCH_SIZE) {
            let mut form = multipart::Form::new();
            for (index, local_path) in chunk.iter().enumerate() {
                let relative = local_path.strip_prefix(local_workspace).map_err(|error| {
                    OptimizerError::Proposer(format!(
                        "cannot compute relative workspace path for {:?}: {error}",
                        local_path
                    ))
                })?;
                let remote_path = remote_join(remote_workspace, relative);
                form = form.text(format!("files[{index}].path"), remote_path.clone());
                let bytes = fs::read(local_path)
                    .map_err(|source| OptimizerError::io(local_path, source))?;
                form = form.part(
                    format!("files[{index}].file"),
                    multipart::Part::bytes(bytes).file_name(remote_path),
                );
            }
            let path = "/files/bulk-upload";
            let url = self.toolbox_path(path);
            let response = self
                .client
                .post(url)
                .headers(header_map(&self.headers)?)
                .multipart(form)
                .send()?;
            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().unwrap_or_default();
                return Err(OptimizerError::Proposer(format!(
                    "daytona workspace upload failed status={status}: {body}"
                )));
            }
        }
        Ok(())
    }

    fn get_json(&self, path: &str) -> Result<Value> {
        let url = self.toolbox_path(path);
        let response = self
            .client
            .get(url)
            .headers(header_map(&self.headers)?)
            .send()?;
        json_response(path, response)
    }

    fn post_json(&self, path: &str, body: &Value) -> Result<Value> {
        let url = self.toolbox_path(path);
        let response = self
            .client
            .post(url)
            .headers(header_map(&self.headers)?)
            .json(body)
            .send()?;
        json_response(path, response)
    }

    fn toolbox_path(&self, path: &str) -> String {
        format!(
            "{}/{sandbox}{path}",
            self.toolbox_url,
            sandbox = url_fragment(&self.sandbox_id)
        )
    }
}

#[derive(Debug)]
struct DaytonaSandbox {
    id: String,
    name: Option<String>,
    state: Option<String>,
    target: Option<String>,
    error_reason: Option<String>,
}

impl DaytonaSandbox {
    fn from_value(value: Value) -> Result<Self> {
        let id = value
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| {
                OptimizerError::Proposer(format!("daytona sandbox response missing id: {value}"))
            })?;
        Ok(Self {
            id,
            name: value
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_string),
            state: value
                .get("state")
                .and_then(Value::as_str)
                .map(str::to_string),
            target: value
                .get("target")
                .and_then(Value::as_str)
                .map(str::to_string),
            error_reason: value
                .get("errorReason")
                .or_else(|| value.get("error_reason"))
                .and_then(Value::as_str)
                .map(str::to_string),
        })
    }
}

fn json_response(path: &str, response: reqwest::blocking::Response) -> Result<Value> {
    if response.status().is_success() {
        let body = response.text()?;
        if body.trim().is_empty() {
            Ok(Value::Null)
        } else {
            Ok(serde_json::from_str(&body)?)
        }
    } else {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        Err(OptimizerError::Proposer(format!(
            "daytona request {path} failed status={status}: {body}"
        )))
    }
}

fn header_map(headers: &BTreeMap<String, String>) -> Result<reqwest::header::HeaderMap> {
    let mut map = reqwest::header::HeaderMap::new();
    for (key, value) in headers {
        let name = reqwest::header::HeaderName::from_bytes(key.as_bytes()).map_err(|error| {
            OptimizerError::Proposer(format!("invalid Daytona header name {key:?}: {error}"))
        })?;
        let value = reqwest::header::HeaderValue::from_str(value).map_err(|error| {
            OptimizerError::Proposer(format!("invalid Daytona header value for {key}: {error}"))
        })?;
        map.insert(name, value);
    }
    Ok(map)
}

fn stage_workspace(original_workspace: &Path, run_id: &str) -> Result<PathBuf> {
    let staging_root = daytona_workspace_root()?;
    let staging_dir = staging_root.join(format!(
        "{}-{}",
        safe_fragment(run_id),
        uuid::Uuid::new_v4().simple()
    ));
    fs::create_dir_all(&staging_dir).map_err(|source| OptimizerError::io(&staging_dir, source))?;
    copy_dir_contents(original_workspace, &staging_dir, true)?;
    Ok(staging_dir)
}

fn sync_workspace_back(staged_workspace: &Path, original_workspace: &Path) -> Result<()> {
    copy_dir_contents(staged_workspace, original_workspace, true)
}

fn cleanup_staged_workspace(staged_workspace: &Path) -> Result<()> {
    if staged_workspace.exists() {
        fs::remove_dir_all(staged_workspace)
            .map_err(|source| OptimizerError::io(staged_workspace, source))?;
    }
    Ok(())
}

fn daytona_workspace_root() -> Result<PathBuf> {
    let home = env::var_os("HOME").ok_or_else(|| {
        OptimizerError::Proposer(
            "daytona proposer staging requires HOME for ~/.cache/synth-gepa-daytona-workspaces"
                .to_string(),
        )
    })?;
    let root = PathBuf::from(home).join(limits::DAYTONA_WORKSPACE_CACHE_DIR);
    fs::create_dir_all(&root).map_err(|source| OptimizerError::io(&root, source))?;
    Ok(root)
}

fn collect_workspace_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_workspace_files_inner(root, &mut files)?;
    Ok(files)
}

fn collect_workspace_files_inner(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(path).map_err(|source| OptimizerError::io(path, source))? {
        let entry = entry.map_err(|source| OptimizerError::io(path, source))?;
        let path = entry.path();
        let metadata = fs::metadata(&path).map_err(|source| OptimizerError::io(&path, source))?;
        if metadata.is_dir() {
            collect_workspace_files_inner(&path, files)?;
        } else if metadata.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

fn copy_dir_contents(source: &Path, destination: &Path, exclude_runtime_auth: bool) -> Result<()> {
    fs::create_dir_all(destination).map_err(|source| OptimizerError::io(destination, source))?;
    for entry in
        fs::read_dir(source).map_err(|read_error| OptimizerError::io(source, read_error))?
    {
        let entry = entry.map_err(|read_error| OptimizerError::io(source, read_error))?;
        let name = entry.file_name();
        if exclude_runtime_auth && excluded_workspace_entry(&name.to_string_lossy()) {
            continue;
        }
        let source_path = entry.path();
        let destination_path = destination.join(&name);
        let metadata = fs::metadata(&source_path)
            .map_err(|metadata_error| OptimizerError::io(&source_path, metadata_error))?;
        if metadata.is_dir() {
            copy_dir_contents(&source_path, &destination_path, exclude_runtime_auth)?;
        } else if metadata.is_file() {
            if let Some(parent) = destination_path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|create_error| OptimizerError::io(parent, create_error))?;
            }
            fs::copy(&source_path, &destination_path)
                .map_err(|copy_error| OptimizerError::io(&destination_path, copy_error))?;
        }
    }
    Ok(())
}

fn excluded_workspace_entry(name: &str) -> bool {
    matches!(
        name,
        ".codex_api_key_home" | ".codex_home" | ".codex_app_server_entrypoint.sh"
    )
}

fn inner_codex_command(proposer: &ProposerConfig) -> Vec<String> {
    if proposer.command.is_empty() {
        vec!["codex".to_string(), "app-server".to_string()]
    } else {
        proposer.command.clone()
    }
}

fn remote_join(base: &str, relative: &Path) -> String {
    let mut value = base.trim_end_matches('/').to_string();
    for component in relative.components() {
        value.push('/');
        value.push_str(&component.as_os_str().to_string_lossy());
    }
    value
}

fn shell_join(parts: &[String]) -> String {
    parts
        .iter()
        .map(|part| shell_quote(part))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | '/' | ':' | '='))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn url_fragment(value: &str) -> String {
    value
        .bytes()
        .flat_map(|byte| {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
                vec![byte as char]
            } else {
                format!("%{byte:02X}").chars().collect::<Vec<_>>()
            }
        })
        .collect()
}

fn safe_fragment(value: &str) -> String {
    let mut output = value
        .chars()
        .filter_map(|ch| {
            if ch.is_ascii_alphanumeric() {
                Some(ch.to_ascii_lowercase())
            } else if matches!(ch, '-' | '_') {
                Some('-')
            } else {
                None
            }
        })
        .take(48)
        .collect::<String>();
    if output.is_empty() {
        output = "run".to_string();
    }
    output
}

fn demux_log(data: &[u8]) -> (Vec<u8>, Vec<u8>) {
    if !data.windows(STDOUT_PREFIX.len()).any(|window| window == STDOUT_PREFIX)
        && !data
            .windows(STDERR_PREFIX.len())
            .any(|window| window == STDERR_PREFIX)
    {
        return (data.to_vec(), Vec::new());
    }
    let mut out = Vec::new();
    let mut err = Vec::new();
    let mut state = 0u8;
    let mut index = 0usize;
    while index < data.len() {
        if data[index..].starts_with(STDOUT_PREFIX) {
            state = 1;
            index += STDOUT_PREFIX.len();
        } else if data[index..].starts_with(STDERR_PREFIX) {
            state = 2;
            index += STDERR_PREFIX.len();
        } else {
            match state {
                1 => out.push(data[index]),
                2 => err.push(data[index]),
                _ => {}
            }
            index += 1;
        }
    }
    (out, err)
}

fn extract_turn_id(message: &Value) -> Option<String> {
    message
        .pointer("/result/turn/id")
        .or_else(|| message.pointer("/result/turnId"))
        .or_else(|| message.pointer("/params/turn/id"))
        .or_else(|| message.pointer("/params/turnId"))
        .or_else(|| message.get("turnId"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn message_matches_turn(message: &Value, turn_id: &str) -> bool {
    message
        .pointer("/params/turn/id")
        .or_else(|| message.pointer("/params/turnId"))
        .or_else(|| message.pointer("/result/turn/id"))
        .or_else(|| message.pointer("/result/turnId"))
        .or_else(|| message.get("turnId"))
        .and_then(Value::as_str)
        == Some(turn_id)
}

fn is_terminal_turn_event(message: &Value) -> bool {
    let method = message.get("method").and_then(Value::as_str).unwrap_or("");
    if matches!(method, "turn/completed" | "turn/failed" | "turn/cancelled") {
        return true;
    }
    message
        .pointer("/params/turn/status")
        .or_else(|| message.pointer("/params/status"))
        .or_else(|| message.pointer("/result/turn/status"))
        .or_else(|| message.pointer("/result/status"))
        .or_else(|| message.get("status"))
        .and_then(Value::as_str)
        .is_some_and(|status| matches!(status, "completed" | "failed" | "cancelled"))
}

fn result_status<T>(result: &Result<T>) -> String {
    match result {
        Ok(_) => "ok".to_string(),
        Err(error) => format!("err({error})"),
    }
}
