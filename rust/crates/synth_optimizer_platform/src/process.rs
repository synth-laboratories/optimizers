use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use crate::config::ContainerConfig;
use crate::error::{OptimizerError, Result};
use crate::http::ContainerClient;
use fs2::FileExt;
use sha2::{Digest, Sha256};

const CONTAINER_TERM_GRACE: Duration = Duration::from_secs(2);
const CONTAINER_KILL_GRACE: Duration = Duration::from_secs(3);
const CONTAINER_LOCK_WAIT_LOG_INTERVAL: Duration = Duration::from_secs(5);

pub struct ManagedContainerProcess {
    child: Option<Child>,
    lock: Option<ContainerLock>,
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
                }));
            }
        }
        let lock = match &config.url {
            Some(url) => Some(acquire_container_lock(url, config.startup_timeout_seconds)?),
            None => None,
        };
        if let Some(url) = &config.url {
            if is_healthy(url) {
                return Ok(Some(Self { child: None, lock }));
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
        command.stdout(Stdio::inherit());
        command.stderr(Stdio::inherit());
        let child = command.spawn().map_err(|source| {
            OptimizerError::io(
                config.cwd.clone().unwrap_or_else(|| PathBuf::from(".")),
                source,
            )
        })?;
        let process = Self {
            child: Some(child),
            lock,
        };
        if let Some(url) = &config.url {
            wait_for_health(url, config.startup_timeout_seconds)?;
        }
        Ok(Some(process))
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
