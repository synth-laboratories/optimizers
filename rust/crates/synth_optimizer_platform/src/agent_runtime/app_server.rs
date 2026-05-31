use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::{OptimizerError, ProposerConfig, Result};

use super::codex_home::prepare_proposer_codex_launch;

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
    pub process_label: String,
}

pub struct CodexAppServerClient {
    child: Child,
    stdin: ChildStdin,
    receiver: Receiver<Result<Value>>,
    buffer: VecDeque<Value>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    auth_home_to_cleanup: Option<PathBuf>,
    next_id: u64,
    sent_messages: Vec<Value>,
    received_messages: Vec<Value>,
}

impl CodexAppServerClient {
    pub fn start(launch: CodexAppServerLaunch<'_>) -> Result<Self> {
        let workspace_dir = fs::canonicalize(launch.workspace_dir)
            .map_err(|source| OptimizerError::io(launch.workspace_dir, source))?;
        let command = if launch.proposer.command.is_empty() {
            vec!["codex".to_string(), "app-server".to_string()]
        } else {
            launch.proposer.command.clone()
        };
        let env_map = env::vars().collect::<BTreeMap<_, _>>();
        let launch_state =
            prepare_proposer_codex_launch(launch.proposer, &workspace_dir, launch.model, env_map)?;
        Self::start_process(CodexAppServerProcessLaunch {
            command,
            current_dir: workspace_dir,
            env_map: launch_state.env_map,
            auth_home_to_cleanup: launch_state.auth_home_to_cleanup,
            process_label: format!("codex app-server model={}", launch.model),
        })
    }

    pub fn start_process(launch: CodexAppServerProcessLaunch) -> Result<Self> {
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
            stdin,
            receiver,
            buffer: VecDeque::new(),
            stderr_tail,
            auth_home_to_cleanup: launch.auth_home_to_cleanup,
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

    pub fn wait_for_response(&mut self, id: u64, timeout: Duration) -> Result<Value> {
        let deadline = Instant::now() + timeout;
        let mut deferred = Vec::new();
        loop {
            let message = self.read_next(deadline)?;
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

    pub fn wait_for_turn_started(&mut self, request_id: u64, timeout: Duration) -> Result<String> {
        let deadline = Instant::now() + timeout;
        let mut deferred = Vec::new();
        loop {
            let message = self.read_next(deadline)?;
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

    pub fn wait_for_turn(&mut self, turn_id: &str, timeout: Duration) -> Result<Value> {
        let deadline = Instant::now() + timeout;
        loop {
            let message = self.read_next(deadline)?;
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

    fn send(&mut self, payload: Value) -> Result<()> {
        serde_json::to_writer(&mut self.stdin, &payload)?;
        self.sent_messages.push(payload);
        self.stdin
            .write_all(b"\n")
            .map_err(|source| OptimizerError::io("codex app-server stdin", source))?;
        self.stdin
            .flush()
            .map_err(|source| OptimizerError::io("codex app-server stdin", source))
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

    fn cleanup_auth_home(&mut self) -> Result<()> {
        if let Some(path) = self.auth_home_to_cleanup.take() {
            if path.exists() {
                fs::remove_dir_all(&path).map_err(|source| OptimizerError::io(&path, source))?;
            }
        }
        Ok(())
    }
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
