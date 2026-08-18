use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::process::{CommandExt, ExitStatusExt};

use crate::config::ContainerConfig;
use crate::error::{OptimizerError, Result};
use crate::http::ContainerClient;
use fs2::FileExt;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const CONTAINER_TERM_GRACE: Duration = Duration::from_secs(2);
const CONTAINER_KILL_GRACE: Duration = Duration::from_secs(3);
const CONTAINER_LOCK_WAIT_LOG_INTERVAL: Duration = Duration::from_secs(5);
const STDERR_TAIL_BYTES: usize = 8_192;

pub struct ManagedContainerProcess {
    child: Option<Child>,
    lock: Option<ContainerLock>,
    stderr_path: Option<PathBuf>,
}

struct ContainerLock {
    file: File,
    path: PathBuf,
}

#[derive(Clone, Debug)]
struct LockHolder {
    pid: String,
    stat: String,
    command: String,
}

impl ManagedContainerProcess {
    pub fn maybe_start(config: &ContainerConfig) -> Result<Option<Self>> {
        if config.command.is_empty() {
            return Ok(None);
        }
        if let Some(url) = &config.url {
            if is_healthy(url) {
                return Ok(Some(Self {
                    child: None,
                    lock: None,
                    stderr_path: None,
                }));
            }
        }
        let lock = match &config.url {
            Some(url) => Some(acquire_container_lock(url, config.startup_timeout_seconds)?),
            None => None,
        };
        if let Some(url) = &config.url {
            if is_healthy(url) {
                return Ok(Some(Self {
                    child: None,
                    lock,
                    stderr_path: None,
                }));
            }
        }
        let program = &config.command[0];
        let args = &config.command[1..];
        let mut command = Command::new(program);
        command.args(args);
        #[cfg(unix)]
        {
            command.process_group(0);
        }
        command.env_remove("VIRTUAL_ENV");
        if let Some(cwd) = &config.cwd {
            command.current_dir(cwd);
        }
        command.stdout(Stdio::null());
        command.stderr(Stdio::piped());
        let mut child = command.spawn().map_err(|source| {
            OptimizerError::io(
                config.cwd.clone().unwrap_or_else(|| PathBuf::from(".")),
                source,
            )
        })?;
        let stderr_path = stderr_capture_path(lock.as_ref(), child.id());
        capture_child_stderr(child.stderr.take(), &stderr_path);
        let process = Self {
            child: Some(child),
            lock,
            stderr_path: Some(stderr_path),
        };
        if let Some(url) = &config.url {
            wait_for_health(url, config.startup_timeout_seconds)?;
        }
        Ok(Some(process))
    }

    #[allow(dead_code)]
    pub fn child_crash_record(&mut self, leased_run_ids: &[String]) -> Option<Value> {
        let child = self.child.as_mut()?;
        let pid = child.id();
        let status = child.try_wait().ok().flatten()?;
        Some(structured_child_crash(
            pid,
            Some(&status),
            None,
            &stderr_tail(self.stderr_path.as_deref()),
            leased_run_ids,
        ))
    }
}

impl Drop for ManagedContainerProcess {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            stop_container_child(child);
        }
        if let Some(lock) = &self.lock {
            let _ = lock.file.unlock();
            let _ = fs::remove_file(&lock.path);
        }
    }
}

#[cfg(unix)]
fn stop_container_child(child: &mut Child) {
    let pgid = child.id() as i32;
    // Negative pid targets the process group created with CommandExt::process_group above.
    unsafe {
        libc::kill(-pgid, libc::SIGTERM);
    }
    if wait_for_child(child, CONTAINER_TERM_GRACE) {
        return;
    }
    let _ = child.kill();
    unsafe {
        libc::kill(-pgid, libc::SIGKILL);
    }
    let _ = wait_for_child(child, CONTAINER_KILL_GRACE);
}

#[cfg(not(unix))]
fn stop_container_child(child: &mut Child) {
    let _ = child.kill();
    let _ = wait_for_child(child, CONTAINER_KILL_GRACE);
}

fn wait_for_child(child: &mut Child, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => return true,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(50)),
            Ok(None) => return false,
            Err(_) => return true,
        }
    }
}

fn acquire_container_lock(url: &str, timeout_seconds: u64) -> Result<ContainerLock> {
    let timeout_seconds = timeout_seconds.max(1);
    let root = std::env::temp_dir().join("synth-optimizers-container-locks");
    fs::create_dir_all(&root).map_err(|source| OptimizerError::io(root.clone(), source))?;
    let path = root.join(format!("{}.lock", stable_lock_id(url)));
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|source| OptimizerError::io(path.clone(), source))?;
    let deadline = Instant::now() + Duration::from_secs(timeout_seconds.max(1));
    let mut last_wait_log = Instant::now()
        .checked_sub(CONTAINER_LOCK_WAIT_LOG_INTERVAL)
        .unwrap_or_else(Instant::now);
    loop {
        match file.try_lock_exclusive() {
            Ok(()) => {
                write_container_lock_metadata(&mut file, url, &path)?;
                return Ok(ContainerLock { file, path });
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                let holders = container_lock_holders(&path);
                let stopped = holders
                    .iter()
                    .filter(|holder| holder_is_stopped(holder))
                    .cloned()
                    .collect::<Vec<_>>();
                if !stopped.is_empty() {
                    return Err(OptimizerError::Container(format!(
                        "container run lock is held by stopped process(es); refusing to wait silently. url={} lock={} holders={}. Resume or terminate the stale optimizer job and retry.",
                        url,
                        path.display(),
                        format_lock_holders(&stopped),
                    )));
                }
                if Instant::now() >= deadline {
                    return Err(OptimizerError::Container(format!(
                        "timed out after {}s waiting for container run lock. url={} lock={} holders={}",
                        timeout_seconds,
                        url,
                        path.display(),
                        format_lock_holders(&holders),
                    )));
                }
                if last_wait_log.elapsed() >= CONTAINER_LOCK_WAIT_LOG_INTERVAL {
                    eprintln!(
                        "waiting for container run lock url={} lock={} holders={}",
                        url,
                        path.display(),
                        format_lock_holders(&holders),
                    );
                    last_wait_log = Instant::now();
                }
                thread::sleep(Duration::from_millis(250));
            }
            Err(source) => return Err(OptimizerError::io(path.clone(), source)),
        }
    }
}

fn write_container_lock_metadata(file: &mut File, url: &str, path: &Path) -> Result<()> {
    file.set_len(0)
        .map_err(|source| OptimizerError::io(path, source))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|source| OptimizerError::io(path, source))?;
    let acquired_at_unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    writeln!(
        file,
        "pid={}\npgid={}\nurl={}\nacquired_at_unix_seconds={}\n",
        std::process::id(),
        current_process_group_id(),
        url,
        acquired_at_unix_seconds,
    )
    .map_err(|source| OptimizerError::io(path, source))?;
    file.flush()
        .map_err(|source| OptimizerError::io(path, source))
}

#[cfg(unix)]
fn current_process_group_id() -> i32 {
    unsafe { libc::getpgrp() }
}

#[cfg(not(unix))]
fn current_process_group_id() -> i32 {
    0
}

fn container_lock_holders(path: &Path) -> Vec<LockHolder> {
    let Ok(output) = Command::new("lsof").arg("-t").arg(path).output() else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let pids = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if pids.is_empty() {
        return Vec::new();
    }
    let Ok(output) = Command::new("ps")
        .args(["-o", "pid=", "-o", "stat=", "-o", "command=", "-p"])
        .arg(pids.join(","))
        .output()
    else {
        return pids
            .into_iter()
            .map(|pid| LockHolder {
                pid,
                stat: "?".to_string(),
                command: "?".to_string(),
            })
            .collect();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(parse_lock_holder_ps_line)
        .collect()
}

fn parse_lock_holder_ps_line(line: &str) -> Option<LockHolder> {
    let mut fields = line.trim().splitn(3, char::is_whitespace);
    let pid = fields.next()?.trim();
    let stat = fields.next()?.trim();
    let command = fields.next().unwrap_or("").trim();
    if pid.is_empty() {
        return None;
    }
    Some(LockHolder {
        pid: pid.to_string(),
        stat: stat.to_string(),
        command: command.to_string(),
    })
}

fn holder_is_stopped(holder: &LockHolder) -> bool {
    holder.stat.contains('T')
}

fn format_lock_holders(holders: &[LockHolder]) -> String {
    if holders.is_empty() {
        return "unknown".to_string();
    }
    holders
        .iter()
        .map(|holder| {
            format!(
                "pid={} stat={} command={}",
                holder.pid, holder.stat, holder.command
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn stable_lock_id(url: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(url.as_bytes());
    let hex = format!("{:x}", digest.finalize());
    hex[..24].to_string()
}

fn is_healthy(url: &str) -> bool {
    ContainerClient::new(url.to_string())
        .and_then(|client| client.health())
        .is_ok()
}

fn wait_for_health(url: &str, timeout_seconds: u64) -> Result<()> {
    let client = ContainerClient::new(url.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(timeout_seconds.max(1));
    loop {
        if client.health().is_ok() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(OptimizerError::Container(format!(
                "container did not become healthy within {}s at {}",
                timeout_seconds, url
            )));
        }
        thread::sleep(Duration::from_millis(250));
    }
}

pub fn structured_child_crash(
    pid: u32,
    status: Option<&ExitStatus>,
    errno: Option<i32>,
    stderr_tail: &str,
    leased_run_ids: &[String],
) -> Value {
    let (exit_status, signal) = status
        .map(exit_status_parts)
        .unwrap_or((None, None));
    json!({
        "schema_version": "synth.gepa_service.crash.v1",
        "cause": "service_crash",
        "reason": "child_death",
        "pid": pid,
        "errno": errno,
        "exit_status": exit_status,
        "signal": signal,
        "stderr_tail": stderr_tail,
        "leased_run_ids": leased_run_ids,
    })
}

fn exit_status_parts(status: &ExitStatus) -> (Option<i32>, Option<i32>) {
    #[cfg(unix)]
    {
        (status.code(), status.signal())
    }
    #[cfg(not(unix))]
    {
        (status.code(), None)
    }
}

fn stderr_capture_path(lock: Option<&ContainerLock>, pid: u32) -> PathBuf {
    if let Some(lock) = lock {
        return lock.path.with_extension(format!("{pid}.stderr"));
    }
    std::env::temp_dir().join(format!("synth-optimizers-child-{pid}.stderr"))
}

fn capture_child_stderr(stderr: Option<std::process::ChildStderr>, path: &Path) {
    let Some(mut stderr) = stderr else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let path = path.to_path_buf();
    thread::spawn(move || {
        let Ok(mut file) = File::create(&path) else {
            return;
        };
        let mut buf = [0u8; 4096];
        loop {
            match stderr.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let _ = file.write_all(&buf[..n]);
                }
            }
        }
        let _ = file.flush();
    });
}

fn stderr_tail(path: Option<&Path>) -> String {
    let Some(path) = path else {
        return String::new();
    };
    let Ok(bytes) = fs::read(path) else {
        return String::new();
    };
    let start = bytes.len().saturating_sub(STDERR_TAIL_BYTES);
    String::from_utf8_lossy(&bytes[start..]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_death_crash_record_captures_exit_and_stderr() {
        let stderr_path = std::env::temp_dir().join(format!(
            "synth-child-stderr-{}-{}.log",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut command = Command::new("sh");
        command
            .args(["-c", "printf 'container boom\\n' >&2; exit 7"])
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let mut child = command.spawn().unwrap();
        let pid = child.id();
        capture_child_stderr(child.stderr.take(), &stderr_path);
        let status = child.wait().unwrap();
        thread::sleep(Duration::from_millis(50));
        let crash = structured_child_crash(
            pid,
            Some(&status),
            None,
            &stderr_tail(Some(&stderr_path)),
            &["run_orphan".to_string()],
        );
        assert_eq!(crash["cause"], "service_crash");
        assert_eq!(crash["reason"], "child_death");
        assert_eq!(crash["pid"], pid);
        assert_eq!(crash["exit_status"], 7);
        assert_eq!(crash["signal"], Value::Null);
        assert!(
            crash["stderr_tail"].as_str().unwrap().contains("container boom"),
            "stderr tail: {}",
            crash["stderr_tail"]
        );
        assert_eq!(crash["leased_run_ids"][0], "run_orphan");
        let _ = fs::remove_file(&stderr_path);
    }
}
