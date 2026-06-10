use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::{OptimizerError, ProposerConfig, Result};

use super::codex_home::{persist_refreshed_chatgpt_codex_auth, prepare_proposer_codex_launch};
use super::jsonrpc_read_window::JsonRpcReadWindow;
use super::substrate::normalize_execution_mode;

pub struct CodexAppServerLaunch<'a> {
    pub proposer: &'a ProposerConfig,
    pub workspace_dir: &'a Path,
    pub model: &'a str,
}

pub struct CodexAppServerProcessLaunch {
    pub command: Vec<String>,
    pub current_dir: PathBuf,
    pub env_map: BTreeMap<String, String>,
    pub auth_home_to_cleanup: Option<PathBuf>,
    pub auth_home_refresh_source: Option<PathBuf>,
    pub process_label: String,
    pub execution_mode: String,
}

enum CodexAppServerTransport {
    Stdio { stdin: ChildStdin },
    WebSocket { stream: TcpStream },
}

pub struct CodexAppServerClient {
    child: Child,
    transport: CodexAppServerTransport,
    receiver: Receiver<Result<Value>>,
    buffer: VecDeque<Value>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    auth_home_to_cleanup: Option<PathBuf>,
    auth_home_refresh_source: Option<PathBuf>,
    next_id: u64,
    sent_messages: Vec<Value>,
    received_messages: Vec<Value>,
}

impl CodexAppServerClient {
    pub fn start(launch: CodexAppServerLaunch<'_>) -> Result<Self> {
        let workspace_dir = fs::canonicalize(launch.workspace_dir)
            .map_err(|source| OptimizerError::io(launch.workspace_dir, source))?;
        let mut command = if launch.proposer.command.is_empty() {
            vec!["codex".to_string(), "app-server".to_string()]
        } else {
            launch.proposer.command.clone()
        };
        append_codex_app_server_config_args(&mut command, launch.proposer);
        let env_map = env::vars().collect::<BTreeMap<_, _>>();
        let launch_state =
            prepare_proposer_codex_launch(launch.proposer, &workspace_dir, launch.model, env_map)?;
        let mut env_map = launch_state.env_map;
        install_codex_arg0_helpers(&command, &mut env_map)?;
        Self::start_process(CodexAppServerProcessLaunch {
            command,
            current_dir: workspace_dir,
            env_map,
            auth_home_to_cleanup: launch_state.auth_home_to_cleanup,
            auth_home_refresh_source: launch_state.auth_home_refresh_source,
            process_label: format!("codex app-server model={}", launch.model),
            execution_mode: launch.proposer.execution_mode.clone(),
        })
    }

    pub fn start_process(launch: CodexAppServerProcessLaunch) -> Result<Self> {
        match normalize_execution_mode(&launch.execution_mode) {
            Some("websocket") => return Self::start_websocket_process(launch),
            Some("stdio") => {}
            _ => {
                return Err(OptimizerError::Proposer(format!(
                    "unsupported codex app-server execution mode {:?}",
                    launch.execution_mode
                )))
            }
        }
        if launch.command.is_empty() {
            return Err(OptimizerError::Proposer(
                "codex app-server command must not be empty".to_string(),
            ));
        }
        let mut cmd = Command::new(&launch.command[0]);
        cmd.args(&launch.command[1..])
            .current_dir(&launch.current_dir)
            .envs(&launch.env_map)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|source| {
            OptimizerError::Proposer(format!(
                "failed to start {} command {:?}: {}",
                launch.process_label, launch.command, source
            ))
        })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            OptimizerError::Proposer("codex app-server stdin unavailable".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            OptimizerError::Proposer("codex app-server stdout unavailable".to_string())
        })?;
        let stderr = child.stderr.take();
        let (sender, receiver) = mpsc::channel();
        let stderr_tail = Arc::new(Mutex::new(VecDeque::new()));
        thread::spawn(move || read_stdout(stdout, sender));
        if let Some(stderr) = stderr {
            let stderr_tail = Arc::clone(&stderr_tail);
            thread::spawn(move || drain_stderr(stderr, stderr_tail));
        }
        Ok(Self {
            child,
            transport: CodexAppServerTransport::Stdio { stdin },
            receiver,
            buffer: VecDeque::new(),
            stderr_tail,
            auth_home_to_cleanup: launch.auth_home_to_cleanup,
            auth_home_refresh_source: launch.auth_home_refresh_source,
            next_id: 1,
            sent_messages: Vec::new(),
            received_messages: Vec::new(),
        })
    }

    fn start_websocket_process(launch: CodexAppServerProcessLaunch) -> Result<Self> {
        if launch.command.is_empty() {
            return Err(OptimizerError::Proposer(
                "codex app-server command must not be empty".to_string(),
            ));
        }
        let port = available_local_port()?;
        let listen_url = format!("ws://127.0.0.1:{port}");
        let mut command = launch.command.clone();
        command.extend(["--listen".to_string(), listen_url.clone()]);
        let mut cmd = Command::new(&command[0]);
        cmd.args(&command[1..])
            .current_dir(&launch.current_dir)
            .envs(&launch.env_map)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|source| {
            OptimizerError::Proposer(format!(
                "failed to start {} websocket command {:?}: {}",
                launch.process_label, command, source
            ))
        })?;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let stderr_tail = Arc::new(Mutex::new(VecDeque::new()));
        if let Some(stdout) = stdout {
            let stderr_tail = Arc::clone(&stderr_tail);
            thread::spawn(move || drain_stderr(stdout, stderr_tail));
        }
        if let Some(stderr) = stderr {
            let stderr_tail = Arc::clone(&stderr_tail);
            thread::spawn(move || drain_stderr(stderr, stderr_tail));
        }
        wait_for_websocket_ready(port, &stderr_tail)?;
        let mut stream = TcpStream::connect(("127.0.0.1", port))
            .map_err(|source| OptimizerError::io("codex app-server websocket connect", source))?;
        stream
            .set_nodelay(true)
            .map_err(|source| OptimizerError::io("codex app-server websocket nodelay", source))?;
        websocket_handshake(&mut stream, port)?;
        let reader = stream
            .try_clone()
            .map_err(|source| OptimizerError::io("codex app-server websocket clone", source))?;
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || read_websocket_messages(reader, sender));
        Ok(Self {
            child,
            transport: CodexAppServerTransport::WebSocket { stream },
            receiver,
            buffer: VecDeque::new(),
            stderr_tail,
            auth_home_to_cleanup: launch.auth_home_to_cleanup,
            auth_home_refresh_source: launch.auth_home_refresh_source,
            next_id: 1,
            sent_messages: Vec::new(),
            received_messages: Vec::new(),
        })
    }

    pub fn sent_messages(&self) -> &[Value] {
        &self.sent_messages
    }

    pub fn received_messages(&self) -> &[Value] {
        &self.received_messages
    }

    pub fn process_id(&self) -> u32 {
        self.child.id()
    }

    pub fn send_request(&mut self, method: &str, params: Value) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(
            serde_json::json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}),
        )?;
        Ok(id)
    }

    pub fn send_notification(&mut self, method: &str, params: Value) -> Result<()> {
        self.send(serde_json::json!({"jsonrpc": "2.0", "method": method, "params": params}))
    }

    pub fn wait_for_response(
        &mut self,
        id: u64,
        timeout: Duration,
        message_stall_timeout: Duration,
    ) -> Result<Value> {
        let window = JsonRpcReadWindow::new(timeout, message_stall_timeout);
        let mut deferred = Vec::new();
        loop {
            if window.overall_expired() {
                return Err(JsonRpcReadWindow::overall_timeout_error(
                    "codex app-server",
                    &self.diagnostic_tail_suffix(),
                ));
            }
            let read_deadline = window.per_read_deadline();
            let message = match self.read_next(read_deadline) {
                Ok(message) => message,
                Err(error) => {
                    return Err(window.map_read_error(
                        "codex app-server",
                        &format!("response to request {id}"),
                        read_deadline,
                        &self.diagnostic_tail_suffix(),
                        error,
                    ));
                }
            };
            if message.get("id").and_then(Value::as_u64) == Some(id)
                && message.get("method").is_none()
            {
                if let Some(error) = message.get("error") {
                    return Err(OptimizerError::Proposer(format!(
                        "codex app-server request {id} failed: {error}"
                    )));
                }
                self.restore_deferred(deferred);
                return Ok(message);
            }
            deferred.push(message);
        }
    }

    pub fn wait_for_turn_started(
        &mut self,
        request_id: u64,
        timeout: Duration,
        message_stall_timeout: Duration,
    ) -> Result<String> {
        let window = JsonRpcReadWindow::new(timeout, message_stall_timeout);
        let mut deferred = Vec::new();
        loop {
            if window.overall_expired() {
                return Err(JsonRpcReadWindow::overall_timeout_error(
                    "codex app-server",
                    &self.diagnostic_tail_suffix(),
                ));
            }
            let read_deadline = window.per_read_deadline();
            let message = match self.read_next(read_deadline) {
                Ok(message) => message,
                Err(error) => {
                    return Err(window.map_read_error(
                        "codex app-server",
                        "turn/started",
                        read_deadline,
                        &self.diagnostic_tail_suffix(),
                        error,
                    ));
                }
            };
            if message.get("id").and_then(Value::as_u64) == Some(request_id)
                && message.get("method").is_none()
            {
                if let Some(error) = message.get("error") {
                    return Err(OptimizerError::Proposer(format!(
                        "codex app-server turn/start request failed: {error}"
                    )));
                }
                let turn_id = extract_turn_id(&message).ok_or_else(|| {
                    OptimizerError::Proposer(format!(
                        "codex app-server turn/start response missing turn id: {message}"
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

    pub fn wait_for_turn(
        &mut self,
        turn_id: &str,
        timeout: Duration,
        message_stall_timeout: Duration,
    ) -> Result<Value> {
        let window = JsonRpcReadWindow::new(timeout, message_stall_timeout);
        loop {
            if window.overall_expired() {
                return Err(JsonRpcReadWindow::overall_timeout_error(
                    "codex app-server",
                    &self.diagnostic_tail_suffix(),
                ));
            }
            let read_deadline = window.per_read_deadline();
            let message = match self.read_next(read_deadline) {
                Ok(message) => message,
                Err(error) => {
                    return Err(window.map_read_error(
                        "codex app-server",
                        "turn/completed|turn/failed|turn/interrupted",
                        read_deadline,
                        &self.diagnostic_tail_suffix(),
                        error,
                    ));
                }
            };
            let method = message
                .get("method")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let matching_turn = match message_turn_id(&message) {
                Some(observed) => observed == turn_id,
                None => true,
            };
            if matches!(
                method,
                "turn/completed" | "turn/failed" | "turn/interrupted"
            ) && matching_turn
            {
                return Ok(message);
            }
        }
    }

    pub fn terminate(&mut self) -> Result<()> {
        if self
            .child
            .try_wait()
            .map_err(|source| {
                OptimizerError::Proposer(format!("failed to inspect codex app-server: {source}"))
            })?
            .is_some()
        {
            return self.cleanup_auth_home();
        }
        self.child.kill().map_err(|source| {
            OptimizerError::Proposer(format!("failed to stop codex app-server: {source}"))
        })?;
        let _ = self.child.wait();
        self.cleanup_auth_home()
    }

    pub fn persist_refreshed_auth_home(&self) -> Result<bool> {
        let (Some(staged_home), Some(source_home)) = (
            self.auth_home_to_cleanup.as_ref(),
            self.auth_home_refresh_source.as_ref(),
        ) else {
            return Ok(false);
        };
        persist_refreshed_chatgpt_codex_auth(staged_home, source_home)
    }

    fn send(&mut self, payload: Value) -> Result<()> {
        self.sent_messages.push(payload);
        match &mut self.transport {
            CodexAppServerTransport::Stdio { stdin } => {
                let payload = self.sent_messages.last().expect("sent payload exists");
                serde_json::to_writer(&mut *stdin, payload)?;
                stdin
                    .write_all(b"\n")
                    .map_err(|source| OptimizerError::io("codex app-server stdin", source))?;
                stdin
                    .flush()
                    .map_err(|source| OptimizerError::io("codex app-server stdin", source))
            }
            CodexAppServerTransport::WebSocket { stream } => {
                let text =
                    serde_json::to_string(self.sent_messages.last().expect("sent payload exists"))?;
                write_websocket_text(stream, text.as_bytes())
            }
        }
    }

    fn restore_deferred(&mut self, deferred: Vec<Value>) {
        for message in deferred.into_iter().rev() {
            self.buffer.push_front(message);
        }
    }

    fn read_next(&mut self, deadline: Instant) -> Result<Value> {
        let now = Instant::now();
        if now >= deadline {
            return Err(OptimizerError::Proposer(format!(
                "codex app-server timed out waiting for response{}",
                self.stderr_tail_suffix()
            )));
        }
        if let Some(message) = self.buffer.pop_front() {
            return Ok(message);
        }
        match self.receiver.recv_timeout(deadline - now) {
            Ok(result) => match result {
                Ok(message) => {
                    self.received_messages.push(message.clone());
                    Ok(message)
                }
                Err(error) => Err(error),
            },
            Err(RecvTimeoutError::Timeout) => Err(OptimizerError::Proposer(format!(
                "codex app-server timed out waiting for response{}",
                self.stderr_tail_suffix()
            ))),
            Err(RecvTimeoutError::Disconnected) => Err(OptimizerError::Proposer(format!(
                "codex app-server stdout closed{}",
                self.stderr_tail_suffix()
            ))),
        }
    }

    fn stderr_tail_suffix(&self) -> String {
        let Ok(tail) = self.stderr_tail.lock() else {
            return String::new();
        };
        if tail.is_empty() {
            return String::new();
        }
        format!(
            "; stderr_tail={}",
            tail.iter().cloned().collect::<Vec<_>>().join("").trim()
        )
    }

    fn diagnostic_tail_suffix(&self) -> String {
        let stderr_tail = self.stderr_tail_suffix();
        let mut parts = Vec::new();
        if !stderr_tail.is_empty() {
            parts.push(stderr_tail);
        }
        if !self.received_messages.is_empty() {
            let tail = self
                .received_messages
                .iter()
                .rev()
                .take(8)
                .map(message_summary)
                .collect::<Vec<_>>();
            parts.push(format!("; received_tail={}", tail.join(" <- ")));
        }
        parts.join("")
    }

    fn cleanup_auth_home(&mut self) -> Result<()> {
        if let Some(path) = self.auth_home_to_cleanup.take() {
            if path.exists() {
                fs::remove_dir_all(&path).map_err(|source| OptimizerError::io(&path, source))?;
            }
        }
        Ok(())
    }
}

fn message_summary(message: &Value) -> String {
    if let Some(method) = message.get("method").and_then(Value::as_str) {
        let turn_id = message_turn_id(message).unwrap_or_default();
        if turn_id.is_empty() {
            return method.to_string();
        }
        return format!("{method}[turn={turn_id}]");
    }
    if let Some(id) = message.get("id").and_then(Value::as_u64) {
        if message.get("error").is_some() {
            return format!("response#{id}:error");
        }
        return format!("response#{id}");
    }
    "message".to_string()
}

pub fn ensure_turn_completed(message: &Value) -> Result<()> {
    let method = message
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if method == "turn/completed" {
        let status = message
            .pointer("/params/turn/status")
            .and_then(Value::as_str)
            .unwrap_or("completed");
        if status == "completed" {
            return Ok(());
        }
    }
    Err(OptimizerError::Proposer(format!(
        "codex app-server turn did not complete: {message}"
    )))
}

pub fn extract_thread_id(message: &Value) -> Option<String> {
    message
        .pointer("/result/thread/id")
        .or_else(|| message.pointer("/result/threadId"))
        .or_else(|| message.pointer("/params/thread/id"))
        .or_else(|| message.pointer("/params/threadId"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn extract_turn_id(message: &Value) -> Option<String> {
    message
        .pointer("/result/turn/id")
        .or_else(|| message.pointer("/result/turnId"))
        .or_else(|| message.pointer("/params/turn/id"))
        .or_else(|| message.pointer("/params/turnId"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn message_turn_id(message: &Value) -> Option<String> {
    extract_turn_id(message)
}

fn append_codex_app_server_config_args(command: &mut Vec<String>, proposer: &ProposerConfig) {
    let Some(service_tier) = proposer.service_tier.as_deref() else {
        return;
    };
    if service_tier.trim().eq_ignore_ascii_case("fast") {
        command.extend([
            "-c".to_string(),
            "service_tier=\"fast\"".to_string(),
            "-c".to_string(),
            "features.fast_mode=true".to_string(),
        ]);
    }
}

fn install_codex_arg0_helpers(
    command: &[String],
    env_map: &mut BTreeMap<String, String>,
) -> Result<()> {
    let Some(native_codex) = resolve_native_codex_binary(command, env_map)? else {
        return Ok(());
    };
    let codex_home = env_map
        .get("CODEX_HOME")
        .map(PathBuf::from)
        .ok_or_else(|| {
            OptimizerError::Proposer("CODEX_HOME missing for codex app-server".to_string())
        })?;
    let helper_dir = codex_home
        .join("tmp")
        .join("arg0")
        .join(format!("codex-arg0{}", uuid::Uuid::new_v4().simple()));
    fs::create_dir_all(&helper_dir).map_err(|source| OptimizerError::io(&helper_dir, source))?;
    fs::write(helper_dir.join(".lock"), b"")
        .map_err(|source| OptimizerError::io(helper_dir.join(".lock"), source))?;
    for helper in ["codex-execve-wrapper", "apply_patch", "applypatch"] {
        let helper_path = helper_dir.join(helper);
        create_codex_helper_link(&native_codex, &helper_path)?;
    }
    prepend_path_entry(env_map, &helper_dir);
    Ok(())
}

fn resolve_native_codex_binary(
    command: &[String],
    env_map: &BTreeMap<String, String>,
) -> Result<Option<PathBuf>> {
    let Some(command_name) = command.first().map(String::as_str) else {
        return Ok(None);
    };
    if !command_looks_like_codex(command_name) {
        return Ok(None);
    }
    let command_path = resolve_command_path(command_name, env_map);
    if let Some(path) = command_path
        .as_ref()
        .filter(|path| native_codex_candidate(path))
    {
        return Ok(Some(path.clone()));
    }
    let mut package_roots = Vec::new();
    if let Some(root) = env_map
        .get("CODEX_MANAGED_PACKAGE_ROOT")
        .filter(|value| !value.trim().is_empty())
    {
        package_roots.push(PathBuf::from(root));
    }
    if let Some(path) = command_path.as_ref() {
        if let Some(root) = package_root_from_codex_js(path) {
            package_roots.push(root);
        }
    }
    for root in package_roots {
        if let Some(path) = native_codex_from_package_root(&root) {
            return Ok(Some(path));
        }
    }
    Err(OptimizerError::Proposer(format!(
        "could not resolve native Codex binary for app-server command {command:?}; unified exec tools require codex arg0 helpers"
    )))
}

fn command_looks_like_codex(command_name: &str) -> bool {
    Path::new(command_name)
        .file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|name| name == "codex" || name == "codex.js")
}

fn resolve_command_path(command_name: &str, env_map: &BTreeMap<String, String>) -> Option<PathBuf> {
    let path = Path::new(command_name);
    if path.components().count() > 1 {
        return fs::canonicalize(path)
            .ok()
            .or_else(|| Some(path.to_path_buf()));
    }
    let path_var = env_map.get("PATH")?;
    for entry in env::split_paths(path_var) {
        let candidate = entry.join(command_name);
        if candidate.is_file() {
            return fs::canonicalize(&candidate).ok().or(Some(candidate));
        }
    }
    None
}

fn native_codex_candidate(path: &Path) -> bool {
    path.file_name().and_then(|value| value.to_str()) == Some("codex")
        && path.extension().and_then(|value| value.to_str()) != Some("js")
        && path.is_file()
}

fn package_root_from_codex_js(path: &Path) -> Option<PathBuf> {
    if path.file_name().and_then(|value| value.to_str()) != Some("codex.js") {
        return None;
    }
    path.parent()?.parent().map(Path::to_path_buf)
}

fn native_codex_from_package_root(root: &Path) -> Option<PathBuf> {
    let triple = platform_target_triple()?;
    let package = platform_package_name()?;
    [
        root.join("vendor").join(triple).join("bin").join("codex"),
        root.join("node_modules")
            .join(package)
            .join("vendor")
            .join(triple)
            .join("bin")
            .join("codex"),
    ]
    .into_iter()
    .find(|path| native_codex_candidate(path))
}

fn platform_target_triple() -> Option<&'static str> {
    match (env::consts::OS, env::consts::ARCH) {
        ("macos", "aarch64") => Some("aarch64-apple-darwin"),
        ("macos", "x86_64") => Some("x86_64-apple-darwin"),
        ("linux", "aarch64") => Some("aarch64-unknown-linux-musl"),
        ("linux", "x86_64") => Some("x86_64-unknown-linux-musl"),
        _ => None,
    }
}

fn platform_package_name() -> Option<&'static str> {
    match (env::consts::OS, env::consts::ARCH) {
        ("macos", "aarch64") => Some("@openai/codex-darwin-arm64"),
        ("macos", "x86_64") => Some("@openai/codex-darwin-x64"),
        ("linux", "aarch64") => Some("@openai/codex-linux-arm64"),
        ("linux", "x86_64") => Some("@openai/codex-linux-x64"),
        _ => None,
    }
}

#[cfg(unix)]
fn create_codex_helper_link(native_codex: &Path, helper_path: &Path) -> Result<()> {
    std::os::unix::fs::symlink(native_codex, helper_path)
        .map_err(|source| OptimizerError::io(helper_path, source))
}

#[cfg(not(unix))]
fn create_codex_helper_link(native_codex: &Path, helper_path: &Path) -> Result<()> {
    fs::copy(native_codex, helper_path)
        .map(|_| ())
        .map_err(|source| OptimizerError::io(helper_path, source))
}

fn prepend_path_entry(env_map: &mut BTreeMap<String, String>, entry: &Path) {
    let mut paths = vec![entry.to_path_buf()];
    if let Some(existing) = env_map.get("PATH") {
        paths.extend(env::split_paths(existing));
    }
    if let Ok(joined) = env::join_paths(paths) {
        env_map.insert("PATH".to_string(), joined.to_string_lossy().to_string());
    }
}

fn read_stdout(stdout: impl Read, sender: mpsc::Sender<Result<Value>>) {
    let mut reader = BufReader::new(stdout);
    loop {
        match read_jsonrpc_message(&mut reader) {
            Ok(Some(value)) => {
                if sender.send(Ok(value)).is_err() {
                    return;
                }
            }
            Ok(None) => return,
            Err(error) => {
                let _ = sender.send(Err(error));
                return;
            }
        }
    }
}

fn read_websocket_messages(mut stream: TcpStream, sender: mpsc::Sender<Result<Value>>) {
    loop {
        match read_websocket_text(&mut stream) {
            Ok(Some(text)) => match serde_json::from_str::<Value>(&text) {
                Ok(value) => {
                    if sender.send(Ok(value)).is_err() {
                        return;
                    }
                }
                Err(source) => {
                    let _ = sender.send(Err(source.into()));
                    return;
                }
            },
            Ok(None) => return,
            Err(error) => {
                let _ = sender.send(Err(error));
                return;
            }
        }
    }
}

fn drain_stderr(stderr: impl Read, tail: Arc<Mutex<VecDeque<String>>>) {
    let mut reader = BufReader::new(stderr);
    let mut line = String::new();
    while reader.read_line(&mut line).unwrap_or(0) > 0 {
        if let Ok(mut tail) = tail.lock() {
            if tail.len() >= 50 {
                tail.pop_front();
            }
            tail.push_back(line.clone());
        }
        line.clear();
    }
}

fn read_jsonrpc_message(reader: &mut BufReader<impl Read>) -> Result<Option<Value>> {
    let mut line = String::new();
    loop {
        line.clear();
        let bytes = reader
            .read_line(&mut line)
            .map_err(|source| OptimizerError::io("codex app-server stdout", source))?;
        if bytes == 0 {
            return Ok(None);
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            return Ok(Some(serde_json::from_str(trimmed)?));
        }
        let mut headers = BTreeMap::new();
        if let Some((key, value)) = trimmed.split_once(':') {
            headers.insert(key.trim().to_ascii_lowercase(), value.trim().to_string());
        }
        loop {
            line.clear();
            let bytes = reader
                .read_line(&mut line)
                .map_err(|source| OptimizerError::io("codex app-server stdout", source))?;
            if bytes == 0 {
                return Ok(None);
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                break;
            }
            if let Some((key, value)) = trimmed.split_once(':') {
                headers.insert(key.trim().to_ascii_lowercase(), value.trim().to_string());
            }
        }
        let raw_len = headers.get("content-length").ok_or_else(|| {
            OptimizerError::Proposer("codex app-server message missing Content-Length".to_string())
        })?;
        let len = raw_len.parse::<usize>().map_err(|source| {
            OptimizerError::Proposer(format!(
                "invalid codex app-server Content-Length {raw_len}: {source}"
            ))
        })?;
        let mut payload = vec![0u8; len];
        reader
            .read_exact(&mut payload)
            .map_err(|source| OptimizerError::io("codex app-server stdout", source))?;
        return Ok(Some(serde_json::from_slice(&payload)?));
    }
}

fn available_local_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .map_err(|source| OptimizerError::io("codex app-server websocket port", source))?;
    let port = listener
        .local_addr()
        .map_err(|source| OptimizerError::io("codex app-server websocket port", source))?
        .port();
    drop(listener);
    Ok(port)
}

fn wait_for_websocket_ready(port: u16, stderr_tail: &Arc<Mutex<VecDeque<String>>>) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(10);
    let url = format!("http://127.0.0.1:{port}/readyz");
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()?;
    while Instant::now() < deadline {
        if client
            .get(&url)
            .send()
            .is_ok_and(|response| response.status().is_success())
        {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    let tail = stderr_tail
        .lock()
        .ok()
        .map(|tail| {
            tail.iter()
                .cloned()
                .collect::<Vec<_>>()
                .join("")
                .trim()
                .to_string()
        })
        .unwrap_or_default();
    Err(OptimizerError::Proposer(format!(
        "codex app-server websocket listener did not become ready on {url}; stderr_tail={tail}"
    )))
}

fn websocket_handshake(stream: &mut TcpStream, port: u16) -> Result<()> {
    let request = format!(
        "GET / HTTP/1.1\r\n\
         Host: 127.0.0.1:{port}\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\
         Sec-WebSocket-Version: 13\r\n\
         \r\n"
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|source| OptimizerError::io("codex app-server websocket handshake", source))?;
    stream
        .flush()
        .map_err(|source| OptimizerError::io("codex app-server websocket handshake", source))?;
    let mut response = Vec::new();
    let mut byte = [0u8; 1];
    while !response.ends_with(b"\r\n\r\n") {
        stream
            .read_exact(&mut byte)
            .map_err(|source| OptimizerError::io("codex app-server websocket handshake", source))?;
        response.push(byte[0]);
        if response.len() > 8192 {
            return Err(OptimizerError::Proposer(
                "codex app-server websocket handshake response exceeded 8 KiB".to_string(),
            ));
        }
    }
    let response_text = String::from_utf8_lossy(&response);
    if !response_text.starts_with("HTTP/1.1 101 ") {
        return Err(OptimizerError::Proposer(format!(
            "codex app-server websocket handshake failed: {}",
            response_text.trim()
        )));
    }
    Ok(())
}

fn read_websocket_text(stream: &mut TcpStream) -> Result<Option<String>> {
    loop {
        let mut header = [0u8; 2];
        match stream.read_exact(&mut header) {
            Ok(()) => {}
            Err(source) if source.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(source) => {
                return Err(OptimizerError::io(
                    "codex app-server websocket frame",
                    source,
                ));
            }
        }
        let opcode = header[0] & 0x0f;
        let masked = header[1] & 0x80 != 0;
        let mut len = u64::from(header[1] & 0x7f);
        if len == 126 {
            let mut extended = [0u8; 2];
            stream
                .read_exact(&mut extended)
                .map_err(|source| OptimizerError::io("codex app-server websocket frame", source))?;
            len = u64::from(u16::from_be_bytes(extended));
        } else if len == 127 {
            let mut extended = [0u8; 8];
            stream
                .read_exact(&mut extended)
                .map_err(|source| OptimizerError::io("codex app-server websocket frame", source))?;
            len = u64::from_be_bytes(extended);
        }
        if len > 16 * 1024 * 1024 {
            return Err(OptimizerError::Proposer(
                "codex app-server websocket frame exceeds 16 MiB".to_string(),
            ));
        }
        let mask = if masked {
            let mut mask = [0u8; 4];
            stream
                .read_exact(&mut mask)
                .map_err(|source| OptimizerError::io("codex app-server websocket mask", source))?;
            Some(mask)
        } else {
            None
        };
        let mut payload = vec![0u8; len as usize];
        stream
            .read_exact(&mut payload)
            .map_err(|source| OptimizerError::io("codex app-server websocket payload", source))?;
        if let Some(mask) = mask {
            for (idx, byte) in payload.iter_mut().enumerate() {
                *byte ^= mask[idx % 4];
            }
        }
        match opcode {
            0x1 => {
                return String::from_utf8(payload).map(Some).map_err(|source| {
                    OptimizerError::Proposer(format!(
                        "codex app-server websocket text was not UTF-8: {source}"
                    ))
                });
            }
            0x8 => return Ok(None),
            0x9 => write_websocket_frame(stream, 0xA, &payload)?,
            0xA => {}
            other => {
                return Err(OptimizerError::Proposer(format!(
                    "unsupported codex app-server websocket opcode {other}"
                )));
            }
        }
    }
}

fn write_websocket_text(stream: &mut TcpStream, payload: &[u8]) -> Result<()> {
    write_websocket_frame(stream, 0x1, payload)
}

fn write_websocket_frame(stream: &mut TcpStream, opcode: u8, payload: &[u8]) -> Result<()> {
    let mut frame = Vec::with_capacity(payload.len() + 14);
    frame.push(0x80 | (opcode & 0x0f));
    let len = payload.len();
    if len < 126 {
        frame.push(0x80 | len as u8);
    } else if len <= u16::MAX as usize {
        frame.push(0x80 | 126);
        frame.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        frame.push(0x80 | 127);
        frame.extend_from_slice(&(len as u64).to_be_bytes());
    }
    let mask = [0x13, 0x37, 0x42, 0x99];
    frame.extend_from_slice(&mask);
    for (idx, byte) in payload.iter().enumerate() {
        frame.push(*byte ^ mask[idx % 4]);
    }
    stream
        .write_all(&frame)
        .map_err(|source| OptimizerError::io("codex app-server websocket write", source))?;
    stream
        .flush()
        .map_err(|source| OptimizerError::io("codex app-server websocket write", source))
}
