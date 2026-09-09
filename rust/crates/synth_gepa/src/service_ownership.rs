use std::fs::{self, File, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, OnceLock,
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use fs2::FileExt;
use serde_json::{json, Value};
use sha2::{Digest as Sha2Digest, Sha256};
use synth_optimizer_platform::{OptimizerError, Result};
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use super::GepaServiceConfig;
use crate::{absolute_path, gepa_home_dir, rfc3339_now};

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);
/// Ownership staleness window. A constant on purpose: the former
/// `SYNTH_GEPA_HEARTBEAT_STALE_SECS` knob was a second authority over who owns
/// the service, and nothing but this file may decide that.
const HEARTBEAT_STALE: Duration = Duration::from_secs(10);
const LOCK_RETRY: Duration = Duration::from_millis(50);
const LOCK_RETRY_BUDGET: Duration = Duration::from_secs(2);

/// Ownership protocol carried in every heartbeat and echoed by `/health` and
/// `/v1/optimizer/capabilities` so Workshop can pin it.
///
/// - 1: pid + `last_seen`; a lock winner signalled whatever pid the old
///   heartbeat named.
/// - 2: pid + `start_identity` + `exe_digest` (+ `instance_id`); a peer is
///   only a peer when all three match, a mismatch is quarantined by rename and
///   never signalled.
pub const OWNERSHIP_PROTOCOL: u8 = 2;

/// What makes a pid *this* process and not a reused number: the kernel's
/// start time for the pid and the digest of the binary we are running.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessIdentity {
    pub pid: u32,
    pub start_identity: Option<String>,
    pub exe_digest: Option<String>,
}

impl ProcessIdentity {
    /// Identity of the calling process. Start identity and exe digest are
    /// computed once per process.
    pub fn current() -> Self {
        static START: OnceLock<Option<String>> = OnceLock::new();
        let pid = std::process::id();
        Self {
            pid,
            start_identity: START
                .get_or_init(|| process_start_identity(pid))
                .clone(),
            exe_digest: current_exe_digest(),
        }
    }

    /// Identity a heartbeat claims for its writer.
    pub fn from_heartbeat(payload: &Value) -> Self {
        Self {
            pid: heartbeat_pid(payload),
            start_identity: payload
                .get("start_identity")
                .and_then(Value::as_str)
                .map(str::to_string),
            exe_digest: payload
                .get("exe_digest")
                .and_then(Value::as_str)
                .map(str::to_string),
        }
    }

    /// True only when the pid is alive *and* the live process has the start
    /// identity and binary this record claims. A record without identity
    /// (protocol 1) never matches. `EPERM` is not alive.
    pub fn matches_live_process(&self) -> bool {
        if self.pid == 0 || !pid_is_alive(self.pid) {
            return false;
        }
        let (Some(start), Some(exe)) = (&self.start_identity, &self.exe_digest) else {
            return false;
        };
        process_start_identity(self.pid).as_deref() == Some(start.as_str())
            && current_exe_digest().as_deref() == Some(exe.as_str())
    }

    pub fn to_json(&self) -> Value {
        json!({
            "pid": self.pid,
            "start_identity": self.start_identity,
            "exe_digest": self.exe_digest,
        })
    }
}

/// Kernel start identity for `pid`, as an opaque string.
/// macOS: `ps -p <pid> -o lstart=`. Linux: `/proc/<pid>/stat` field 22
/// (`starttime`, clock ticks since boot). `None` when the pid is gone or the
/// platform has no derivation — which reads as "not a match", never "alive".
pub fn process_start_identity(pid: u32) -> Option<String> {
    if pid == 0 {
        return None;
    }
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("ps")
            .arg("-p")
            .arg(pid.to_string())
            .arg("-o")
            .arg("lstart=")
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        (!text.is_empty()).then_some(text)
    }
    #[cfg(target_os = "linux")]
    {
        let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        // `comm` (field 2) may contain spaces; everything after the closing
        // paren starts at field 3, so field 22 is index 19.
        let (_, rest) = stat.rsplit_once(')')?;
        rest.split_whitespace().nth(19).map(str::to_string)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

/// `sha256:<hex>` of the running executable, computed once per process.
pub fn current_exe_digest() -> Option<String> {
    static DIGEST: OnceLock<Option<String>> = OnceLock::new();
    DIGEST
        .get_or_init(|| {
            let exe = std::env::current_exe().ok()?;
            let bytes = fs::read(exe).ok()?;
            Some(format!("sha256:{:x}", Sha256::digest(bytes)))
        })
        .clone()
}

/// OS-owned lease for one GEPA service identity keyed by
/// `(workshop_instance_id, db_path)`.
pub struct ServiceOwnershipGuard {
    lock: Option<File>,
    lock_path: PathBuf,
    heartbeat_path: PathBuf,
    pid: u32,
    stop: Arc<AtomicBool>,
    writer: Mutex<Option<JoinHandle<()>>>,
    service_url: Arc<Mutex<String>>,
    pub adopted_crash: Option<Value>,
}

impl ServiceOwnershipGuard {
    pub fn heartbeat_path(&self) -> &Path {
        &self.heartbeat_path
    }

    pub fn lock_path(&self) -> &Path {
        &self.lock_path
    }

    /// Stop the heartbeat writer and wait for it to exit, so a test that
    /// rewrites the heartbeat afterwards cannot race a write already in
    /// flight. Production drops never join: the thread exits on its own
    /// after its current sleep.
    #[cfg(test)]
    pub fn stop_heartbeat_writer(&self) {
        self.stop.store(true, Ordering::Relaxed);
        let handle = self.writer.lock().ok().and_then(|mut writer| writer.take());
        if let Some(handle) = handle {
            let _ = handle.join();
        }
    }
}

impl Drop for ServiceOwnershipGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Detach, never join: the writer exits on its own after its current
        // sleep, and a drop must not wait a heartbeat interval for it.
        if let Ok(writer) = self.writer.get_mut() {
            drop(writer.take());
        }
        if !heartbeat_pid_matches(&self.heartbeat_path, self.pid) {
            if let Some(lock) = self.lock.take() {
                let _ = lock.unlock();
            }
            return;
        }
        let _ = fs::remove_file(&self.heartbeat_path);
        if let Some(lock) = self.lock.take() {
            let _ = lock.unlock();
        }
    }
}

pub fn acquire_service_ownership(config: &GepaServiceConfig) -> Result<ServiceOwnershipGuard> {
    acquire_service_ownership_in(gepa_home_dir(), config)
}

pub(crate) fn acquire_service_ownership_in(
    home: PathBuf,
    config: &GepaServiceConfig,
) -> Result<ServiceOwnershipGuard> {
    let services = home.join("services");
    fs::create_dir_all(&services).map_err(|source| OptimizerError::io(&services, source))?;
    let service_id = service_id_for(config);
    let lock_path = services.join(format!("{service_id}.lock"));
    let heartbeat_path = services.join(format!("{service_id}.json"));
    let deadline = OffsetDateTime::now_utc() + LOCK_RETRY_BUDGET;
    let own_pid = std::process::id();
    let mut adopted_crash = None;

    loop {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|source| OptimizerError::io(&lock_path, source))?;
        match file.try_lock_exclusive() {
            Ok(()) => {
                // We hold the lock: whoever wrote the heartbeat no longer does.
                if let Some(peer) = read_heartbeat(&heartbeat_path)? {
                    let identity = ProcessIdentity::from_heartbeat(&peer);
                    if identity.matches_live_process() {
                        // Our own lineage, alive, lost its lock. The only
                        // path that may signal a process.
                        if identity.pid != own_pid {
                            kill_pid(identity.pid);
                            adopted_crash = Some(orphan_sidecar_crash(&peer, identity.pid));
                        }
                    } else {
                        let quarantine =
                            quarantine_peer(&service_id, &heartbeat_path, None, &peer, "lock_free");
                        adopted_crash =
                            Some(stale_heartbeat_crash(&peer, identity.pid, quarantine));
                    }
                }
                return start_owned_guard(
                    file,
                    lock_path,
                    heartbeat_path,
                    config,
                    adopted_crash,
                );
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                match read_heartbeat(&heartbeat_path)? {
                    Some(peer) if peer_is_healthy(&peer) => {
                        return Err(already_running_error(&peer, config));
                    }
                    Some(peer) => {
                        let identity = ProcessIdentity::from_heartbeat(&peer);
                        if identity.matches_live_process() {
                            // Our own lineage holding the lock but not
                            // heartbeating: hung. Signal it and retry.
                            if identity.pid != own_pid {
                                kill_pid(identity.pid);
                                adopted_crash = Some(orphan_sidecar_crash(&peer, identity.pid));
                            }
                        } else {
                            // The lock holder is not the process this record
                            // describes (dead writer, reused pid, other
                            // binary, protocol-1 writer). Move both files
                            // aside — the holder keeps its flock on the
                            // renamed inode and is never signalled — then
                            // take a fresh lock at the canonical path.
                            let quarantine = quarantine_peer(
                                &service_id,
                                &heartbeat_path,
                                Some(&lock_path),
                                &peer,
                                "lock_held",
                            );
                            adopted_crash =
                                Some(stale_heartbeat_crash(&peer, identity.pid, quarantine));
                            drop(file);
                            continue;
                        }
                        if OffsetDateTime::now_utc() > deadline {
                            return Err(OptimizerError::Invariant(format!(
                                "timed out breaking stale GEPA service lock {}",
                                lock_path.display()
                            )));
                        }
                        thread::sleep(LOCK_RETRY);
                    }
                    None => {
                        if OffsetDateTime::now_utc() > deadline {
                            return Err(OptimizerError::Invariant(format!(
                                "GEPA service lock held without heartbeat: {}",
                                lock_path.display()
                            )));
                        }
                        thread::sleep(LOCK_RETRY);
                    }
                }
            }
            Err(source) => return Err(OptimizerError::io(lock_path, source)),
        }
    }
}

fn start_owned_guard(
    lock: File,
    lock_path: PathBuf,
    heartbeat_path: PathBuf,
    config: &GepaServiceConfig,
    adopted_crash: Option<Value>,
) -> Result<ServiceOwnershipGuard> {
    let pid = std::process::id();
    let started_at = rfc3339_now();
    let placeholder_url = service_url_placeholder(&config.bind_addr);
    write_owned_heartbeat(&heartbeat_path, config, &placeholder_url, &started_at)?;
    let stop = Arc::new(AtomicBool::new(false));
    let service_url = Arc::new(Mutex::new(placeholder_url.clone()));
    let thread_path = heartbeat_path.clone();
    let thread_config = config.clone();
    let thread_url = service_url.clone();
    let thread_started_at = started_at;
    let thread_stop = stop.clone();
    let writer = thread::spawn(move || {
        while !thread_stop.load(Ordering::Relaxed) {
            let url = thread_url
                .lock()
                .map(|value| value.clone())
                .unwrap_or_else(|_| "http://127.0.0.1:0".to_string());
            let _ = write_owned_heartbeat(&thread_path, &thread_config, &url, &thread_started_at);
            thread::sleep(HEARTBEAT_INTERVAL);
        }
    });
    Ok(ServiceOwnershipGuard {
        lock: Some(lock),
        lock_path,
        heartbeat_path,
        pid,
        stop,
        writer: Mutex::new(Some(writer)),
        service_url,
        adopted_crash,
    })
}

pub fn refresh_owned_heartbeat(
    guard: &ServiceOwnershipGuard,
    config: &GepaServiceConfig,
    service_url: &str,
    started_at: &str,
) -> Result<()> {
    if !heartbeat_pid_matches(&guard.heartbeat_path, guard.pid) {
        return Ok(());
    }
    if let Ok(mut url) = guard.service_url.lock() {
        *url = service_url.to_string();
    }
    write_owned_heartbeat(&guard.heartbeat_path, config, service_url, started_at)
}

pub fn service_id_for(config: &GepaServiceConfig) -> String {
    service_id_for_parts(
        config.workshop_instance_id.as_deref().unwrap_or(""),
        &config.db_path,
    )
}

pub fn service_id_for_parts(workshop_instance_id: &str, db_path: &Path) -> String {
    let mut hasher = Sha256::new();
    if !workshop_instance_id.is_empty() {
        Sha2Digest::update(&mut hasher, workshop_instance_id.as_bytes());
        Sha2Digest::update(&mut hasher, b"\0");
    }
    Sha2Digest::update(
        &mut hasher,
        absolute_path(db_path).display().to_string().as_bytes(),
    );
    format!("{:x}", Sha2Digest::finalize(hasher))
}

fn write_owned_heartbeat(
    path: &Path,
    config: &GepaServiceConfig,
    service_url: &str,
    started_at: &str,
) -> Result<()> {
    let payload = owned_heartbeat_payload(config, service_url, started_at, Some(rfc3339_now()));
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(&payload)?)
        .map_err(|source| OptimizerError::io(&tmp, source))?;
    fs::rename(&tmp, path).map_err(|source| OptimizerError::io(path, source))
}

/// The process fields Workshop verifies against the child it spawned. One
/// shape for the heartbeat, `/health`, and `/v1/optimizer/capabilities`.
pub fn process_identity_payload(config: &GepaServiceConfig) -> Value {
    let identity = ProcessIdentity::current();
    let mut payload = identity.to_json();
    if let Some(object) = payload.as_object_mut() {
        object.insert(
            "ownership_protocol".to_string(),
            json!(OWNERSHIP_PROTOCOL),
        );
        // Recorded only when Workshop set SYNTH_WORKSHOP_INSTANCE_ID.
        if let Some(instance_id) = config.workshop_instance_id.as_deref() {
            object.insert("instance_id".to_string(), json!(instance_id));
        }
    }
    payload
}

pub fn owned_heartbeat_payload(
    config: &GepaServiceConfig,
    service_url: &str,
    started_at: &str,
    last_seen: Option<String>,
) -> Value {
    let mut payload = json!({
        "kind": "gepa-service",
        "schema": "synth.gepa_service.whoami.v2",
        "version": env!("CARGO_PKG_VERSION"),
        "source_id": service_id_for(config),
        "service_url": service_url,
        "bind": config.bind_addr.clone(),
        "db_path": absolute_path(&config.db_path).display().to_string(),
        "workshop_instance_id": config.workshop_instance_id.clone().unwrap_or_default(),
        "worker_id": config.worker_id.clone(),
        "workers": config.worker_count,
        // Run-request lease length copied through as metadata for operators.
        // Nothing reads it for ownership; ownership is `last_seen` within
        // HEARTBEAT_STALE plus the identity fields below.
        "lease_seconds": config.lease_seconds,
        "started_at": started_at,
        "run_roots": [],
    });
    if let (Some(object), Some(identity)) = (
        payload.as_object_mut(),
        process_identity_payload(config).as_object(),
    ) {
        for (key, value) in identity {
            object.insert(key.clone(), value.clone());
        }
        if let Some(last_seen) = last_seen {
            object.insert("last_seen".to_string(), json!(last_seen));
        }
    }
    payload
}

fn read_heartbeat(path: &Path) -> Result<Option<Value>> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(OptimizerError::io(path, source)),
    };
    if bytes.is_empty() {
        return Ok(None);
    }
    Ok(serde_json::from_slice(&bytes).ok())
}

fn heartbeat_pid(payload: &Value) -> u32 {
    payload
        .get("pid")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .min(u32::MAX as u64) as u32
}

fn heartbeat_pid_matches(path: &Path, pid: u32) -> bool {
    read_heartbeat(path)
        .ok()
        .flatten()
        .is_some_and(|payload| heartbeat_pid(&payload) == pid)
}

fn last_seen_is_fresh(payload: &Value) -> bool {
    let Some(last_seen) = payload.get("last_seen").and_then(Value::as_str) else {
        return false;
    };
    let Ok(timestamp) = OffsetDateTime::parse(last_seen, &Rfc3339) else {
        return false;
    };
    OffsetDateTime::now_utc() - timestamp < HEARTBEAT_STALE
}

/// pid alive ∧ start identity matches ∧ exe digest matches ∧ last_seen fresh.
fn peer_is_healthy(payload: &Value) -> bool {
    ProcessIdentity::from_heartbeat(payload).matches_live_process() && last_seen_is_fresh(payload)
}

/// Move `<id>.json` (and, when the lock is held by a stranger, `<id>.lock`)
/// to `<name>.stale-<pid>`. Never deletes; a prior quarantine of the same pid
/// gets a numeric suffix rather than being overwritten. Logs one line.
fn quarantine_peer(
    service_id: &str,
    heartbeat_path: &Path,
    lock_path: Option<&Path>,
    peer: &Value,
    lock_state: &str,
) -> Value {
    let pid = heartbeat_pid(peer);
    let heartbeat = quarantine_path(heartbeat_path, pid);
    let lock = lock_path.and_then(|path| quarantine_path(path, pid));
    let record = json!({
        "heartbeat": heartbeat.as_ref().map(|path| path.display().to_string()),
        "lock": lock.as_ref().map(|path| path.display().to_string()),
    });
    eprintln!(
        "{}",
        json!({
            "event": "gepa_service.ownership.quarantined",
            "service_id": service_id,
            "peer_pid": pid,
            "peer_start_identity": peer.get("start_identity").cloned().unwrap_or(Value::Null),
            "peer_exe_digest": peer.get("exe_digest").cloned().unwrap_or(Value::Null),
            "peer_ownership_protocol": peer.get("ownership_protocol").cloned().unwrap_or(Value::Null),
            "lock_state": lock_state,
            "quarantined": record,
            "adopted_by_pid": std::process::id(),
        })
    );
    record
}

fn quarantine_path(path: &Path, pid: u32) -> Option<PathBuf> {
    let name = path.file_name()?.to_string_lossy().into_owned();
    let mut target = path.with_file_name(format!("{name}.stale-{pid}"));
    let mut attempt = 1u32;
    while target.exists() {
        target = path.with_file_name(format!("{name}.stale-{pid}-{attempt}"));
        attempt += 1;
    }
    match fs::rename(path, &target) {
        Ok(()) => Some(target),
        Err(_) => None,
    }
}

fn already_running_error(peer: &Value, config: &GepaServiceConfig) -> OptimizerError {
    let pid = heartbeat_pid(peer);
    let service_url = peer
        .get("service_url")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    OptimizerError::AlreadyRunning {
        pid,
        service_url: service_url.clone(),
        db_path: absolute_path(&config.db_path).display().to_string(),
        workshop_instance_id: config.workshop_instance_id.clone().unwrap_or_default(),
        peer: json!({
            "code": "already_running",
            "pid": pid,
            "start_identity": peer.get("start_identity").cloned().unwrap_or(Value::Null),
            "instance_id": peer.get("instance_id").cloned().unwrap_or(Value::Null),
            "ownership_protocol": peer.get("ownership_protocol").cloned().unwrap_or(Value::Null),
            "service_url": service_url,
            "db_path": peer.get("db_path").cloned().unwrap_or(Value::Null),
            "workshop_instance_id": peer.get("workshop_instance_id").cloned().unwrap_or(json!("")),
            "bind": peer.get("bind").cloned().unwrap_or(Value::Null),
            "last_seen": peer.get("last_seen").cloned().unwrap_or(Value::Null),
        }),
    }
}

fn orphan_sidecar_crash(peer: &Value, pid: u32) -> Value {
    json!({
        "schema_version": "synth.gepa_service.crash.v1",
        "cause": "service_crash",
        "reason": "orphan_sidecar",
        "pid": pid,
        "errno": None::<i32>,
        "exit_status": None::<i32>,
        "signal": signal_term(),
        "stderr_tail": "",
        "leased_run_ids": [],
        "peer": peer,
    })
}

fn stale_heartbeat_crash(peer: &Value, pid: u32, quarantined: Value) -> Value {
    json!({
        "schema_version": "synth.gepa_service.crash.v1",
        "cause": "service_crash",
        "reason": "stale_heartbeat",
        "pid": pid,
        "errno": None::<i32>,
        "exit_status": None::<i32>,
        "signal": None::<i32>,
        "stderr_tail": "",
        "leased_run_ids": [],
        "quarantined": quarantined,
        "peer": peer,
    })
}

fn signal_term() -> Option<i32> {
    #[cfg(unix)]
    {
        Some(libc::SIGTERM)
    }
    #[cfg(not(unix))]
    {
        None
    }
}

fn service_url_placeholder(bind_addr: &str) -> String {
    if bind_addr.contains("://") {
        bind_addr.to_string()
    } else {
        format!("http://{bind_addr}")
    }
}

/// `kill(pid, 0) == 0`. `EPERM` means "a process we may not signal exists
/// there" — a reused pid owned by someone else — and is **not** alive for
/// ownership purposes.
#[cfg(unix)]
fn pid_is_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

#[cfg(not(unix))]
fn pid_is_alive(pid: u32) -> bool {
    pid != 0 && pid == std::process::id()
}

/// Only reachable after `ProcessIdentity::matches_live_process` returned
/// true for `pid`: same binary, same start identity, alive.
#[cfg(unix)]
fn kill_pid(pid: u32) {
    if pid == 0 || pid == std::process::id() {
        return;
    }
    unsafe {
        libc::kill(pid as i32, libc::SIGTERM);
    }
    thread::sleep(Duration::from_millis(50));
    if pid_is_alive(pid) {
        unsafe {
            libc::kill(pid as i32, libc::SIGKILL);
        }
    }
}

#[cfg(not(unix))]
fn kill_pid(_pid: u32) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use std::time::Instant;

    const HOLD_HOME: &str = "SYNTH_GEPA_OWNERSHIP_HOLD_HOME";
    const HOLD_DB: &str = "SYNTH_GEPA_OWNERSHIP_HOLD_DB";
    const HOLD_INSTANCE: &str = "SYNTH_GEPA_OWNERSHIP_HOLD_INSTANCE";
    const HOLD_READY: &str = "SYNTH_GEPA_OWNERSHIP_HOLD_READY";

    fn scratch_home(label: &str) -> PathBuf {
        let nanos = OffsetDateTime::now_utc().unix_timestamp_nanos();
        let dir = std::env::temp_dir().join(format!(
            "gepa-own-{label}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn test_config(db_path: PathBuf, instance: &str) -> GepaServiceConfig {
        GepaServiceConfig {
            db_path,
            bind_addr: "127.0.0.1:0".to_string(),
            worker_id: "synth-gepa-service".to_string(),
            lease_seconds: 60,
            worker_count: 2,
            workshop_instance_id: Some(instance.to_string()),
        }
    }

    fn hold_lock_if_requested() -> bool {
        let Ok(home) = std::env::var(HOLD_HOME) else {
            return false;
        };
        let db = PathBuf::from(std::env::var(HOLD_DB).expect("hold db path"));
        let instance = std::env::var(HOLD_INSTANCE).expect("hold instance");
        let ready = PathBuf::from(std::env::var(HOLD_READY).expect("hold ready path"));
        let config = test_config(db, &instance);
        let _guard = acquire_service_ownership_in(PathBuf::from(home), &config)
            .expect("holder must acquire the service lock");
        fs::write(&ready, b"ready").expect("holder ready file");
        loop {
            thread::sleep(Duration::from_secs(1));
        }
    }

    fn wait_for_file(path: &Path, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if path.is_file() {
                return;
            }
            thread::sleep(Duration::from_millis(20));
        }
        panic!("timed out waiting for {}", path.display());
    }

    fn stale_files(services: &Path, service_id: &str, pid: u32) -> (PathBuf, PathBuf) {
        (
            services.join(format!("{service_id}.json.stale-{pid}")),
            services.join(format!("{service_id}.lock.stale-{pid}")),
        )
    }

    #[test]
    fn service_id_is_keyed_by_workshop_instance_and_db_path() {
        let db_a = PathBuf::from("/tmp/gepa-a.sqlite");
        let db_b = PathBuf::from("/tmp/gepa-b.sqlite");
        let left = service_id_for_parts("workshop-a", &db_a);
        let right = service_id_for_parts("workshop-a", &db_a);
        assert_eq!(left, right);
        assert_ne!(left, service_id_for_parts("workshop-b", &db_a));
        assert_ne!(left, service_id_for_parts("workshop-a", &db_b));
        assert_ne!(
            service_id_for_parts("", &db_a),
            service_id_for_parts("workshop-a", &db_a)
        );
    }

    #[test]
    fn heartbeat_carries_protocol_2_identity() {
        let config = test_config(PathBuf::from("/tmp/gepa-identity.sqlite"), "workshop-id");
        let payload = owned_heartbeat_payload(&config, "http://127.0.0.1:9", "2000-01-01T00:00:00Z", None);
        assert_eq!(payload["ownership_protocol"], OWNERSHIP_PROTOCOL);
        assert_eq!(payload["pid"], std::process::id());
        assert_eq!(payload["instance_id"], "workshop-id");
        let start = payload["start_identity"].as_str().expect("start identity present");
        assert!(!start.is_empty());
        assert_eq!(Some(start.to_string()), process_start_identity(std::process::id()));
        let exe = payload["exe_digest"].as_str().expect("exe digest present");
        assert!(exe.starts_with("sha256:"));
        assert!(ProcessIdentity::from_heartbeat(&payload).matches_live_process());

        let unset = GepaServiceConfig {
            workshop_instance_id: None,
            ..config
        };
        let payload = owned_heartbeat_payload(&unset, "http://127.0.0.1:9", "2000-01-01T00:00:00Z", None);
        assert!(payload.get("instance_id").is_none(), "instance_id is recorded only when set");
        let process = process_identity_payload(&unset);
        assert_eq!(process["ownership_protocol"], OWNERSHIP_PROTOCOL);
        assert_eq!(process["pid"], std::process::id());
        assert!(process["start_identity"].is_string());
    }

    #[test]
    fn eperm_is_not_alive() {
        #[cfg(unix)]
        {
            if unsafe { libc::geteuid() } == 0 {
                return; // root may signal pid 1; the EPERM branch is unobservable
            }
            assert!(!pid_is_alive(1), "pid 1 answers EPERM to kill(1, 0) and must not read as alive");
        }
        assert!(!pid_is_alive(0));
        assert!(pid_is_alive(std::process::id()));
    }

    #[test]
    fn healthy_peer_loser_gets_already_running() {
        if hold_lock_if_requested() {
            return;
        }
        let home = scratch_home("race");
        let db = home.join("workspace.sqlite");
        fs::write(&db, b"").unwrap();
        let ready = home.join("holder.ready");
        let instance = "workshop-race";
        let exe = std::env::current_exe().expect("current test exe");
        let mut child = Command::new(&exe)
            .arg("service::service_ownership::tests::healthy_peer_loser_gets_already_running")
            .arg("--exact")
            .arg("--nocapture")
            .env(HOLD_HOME, &home)
            .env(HOLD_DB, &db)
            .env(HOLD_INSTANCE, instance)
            .env(HOLD_READY, &ready)
            .env("RUST_TEST_THREADS", "1")
            .spawn()
            .expect("spawn lock holder");
        wait_for_file(&ready, Duration::from_secs(8));
        let config = test_config(db.clone(), instance);
        let err = match acquire_service_ownership_in(home.clone(), &config) {
            Ok(_guard) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("second owner must fail with already_running");
            }
            Err(err) => err,
        };
        let _ = child.kill();
        let child_pid = child.id();
        let _ = child.wait();
        assert_eq!(err.error_code(), "already_running");
        match err {
            OptimizerError::AlreadyRunning {
                pid,
                service_url,
                db_path,
                workshop_instance_id,
                ref peer,
            } => {
                assert_eq!(pid, child_pid);
                assert!(!service_url.is_empty());
                assert!(db_path.contains("workspace.sqlite"));
                assert_eq!(workshop_instance_id, instance);
                assert_eq!(peer["code"], "already_running");
                assert_eq!(peer["pid"], pid);
                assert_eq!(peer["ownership_protocol"], OWNERSHIP_PROTOCOL);
                assert!(peer["start_identity"].is_string());
            }
            other => panic!("expected AlreadyRunning, got {other:?}"),
        }
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn dead_pid_heartbeat_is_quarantined_and_adopted() {
        let home = scratch_home("stale");
        let db = home.join("workspace.sqlite");
        fs::write(&db, b"").unwrap();
        let config = test_config(db, "workshop-stale");
        let service_id = service_id_for(&config);
        let services = home.join("services");
        fs::create_dir_all(&services).unwrap();
        let heartbeat_path = services.join(format!("{service_id}.json"));
        let dead_pid = i32::MAX as u32;
        let mut payload = owned_heartbeat_payload(
            &config,
            "http://127.0.0.1:9",
            "2000-01-01T00:00:00Z",
            Some(rfc3339_now()),
        );
        payload["pid"] = json!(dead_pid);
        fs::write(
            &heartbeat_path,
            serde_json::to_vec_pretty(&payload).unwrap(),
        )
        .unwrap();

        let guard = acquire_service_ownership_in(home.clone(), &config)
            .expect("dead-pid heartbeat must be adoptable");
        assert!(guard.lock_path().exists());
        let crash = guard
            .adopted_crash
            .as_ref()
            .expect("adopted crash record");
        assert_eq!(crash["cause"], "service_crash");
        assert_eq!(crash["reason"], "stale_heartbeat");
        assert_eq!(crash["pid"], dead_pid);
        let (stale_heartbeat, _) = stale_files(&services, &service_id, dead_pid);
        assert!(stale_heartbeat.is_file(), "heartbeat renamed, not deleted");
        assert_eq!(crash["quarantined"]["heartbeat"], stale_heartbeat.display().to_string());
        let quarantined: Value =
            serde_json::from_slice(&fs::read(&stale_heartbeat).unwrap()).unwrap();
        assert_eq!(quarantined["pid"], dead_pid);
        let own: Value = serde_json::from_slice(&fs::read(&heartbeat_path).unwrap()).unwrap();
        assert_eq!(own["pid"], std::process::id());
        guard.stop_heartbeat_writer();
        drop(guard);
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn foreign_live_pid_holding_the_lock_is_quarantined_not_killed() {
        let home = scratch_home("foreign");
        let db = home.join("workspace.sqlite");
        fs::write(&db, b"").unwrap();
        let config = test_config(db, "workshop-foreign");
        let service_id = service_id_for(&config);
        let services = home.join("services");
        fs::create_dir_all(&services).unwrap();
        let heartbeat_path = services.join(format!("{service_id}.json"));
        let lock_path = services.join(format!("{service_id}.lock"));

        // A live process we did not spawn as a sidecar, with a heartbeat that
        // names its pid but not its start identity (a reused pid).
        let mut sleeper = Command::new("sleep").arg("60").spawn().expect("spawn sleep");
        let foreign_pid = sleeper.id();
        assert!(pid_is_alive(foreign_pid));
        let holder = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .unwrap();
        holder.try_lock_exclusive().expect("test holds the flock");
        let mut payload = owned_heartbeat_payload(
            &config,
            "http://127.0.0.1:9",
            "2000-01-01T00:00:00Z",
            Some(rfc3339_now()),
        );
        payload["pid"] = json!(foreign_pid);
        payload["start_identity"] = json!("not-the-start-identity-of-that-pid");
        fs::write(&heartbeat_path, serde_json::to_vec_pretty(&payload).unwrap()).unwrap();

        let guard = acquire_service_ownership_in(home.clone(), &config)
            .expect("identity mismatch must be adopted, not refused");
        assert!(
            sleeper.try_wait().expect("poll sleep").is_none(),
            "a process whose identity does not match must never be signalled"
        );
        assert!(pid_is_alive(foreign_pid));
        let (stale_heartbeat, stale_lock) = stale_files(&services, &service_id, foreign_pid);
        assert!(stale_heartbeat.is_file(), "heartbeat renamed aside");
        assert!(stale_lock.is_file(), "held lock renamed aside, holder keeps its inode");
        let crash = guard.adopted_crash.as_ref().expect("adopted crash record");
        assert_eq!(crash["reason"], "stale_heartbeat");
        assert_eq!(crash["pid"], foreign_pid);
        assert_eq!(crash["quarantined"]["lock"], stale_lock.display().to_string());
        let own: Value = serde_json::from_slice(&fs::read(&heartbeat_path).unwrap()).unwrap();
        assert_eq!(own["pid"], std::process::id());
        assert_eq!(guard.lock_path(), lock_path.as_path());

        guard.stop_heartbeat_writer();
        drop(guard);
        let _ = sleeper.kill();
        let _ = sleeper.wait();
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn drop_does_not_delete_heartbeat_owned_by_another_pid() {
        let home = scratch_home("drop");
        let db = home.join("workspace.sqlite");
        fs::write(&db, b"").unwrap();
        let config = test_config(db, "workshop-drop");
        let guard = acquire_service_ownership_in(home.clone(), &config).unwrap();
        let heartbeat_path = guard.heartbeat_path().to_path_buf();
        let mut stolen = read_heartbeat(&heartbeat_path)
            .unwrap()
            .expect("owner heartbeat");
        stolen["pid"] = json!(1);
        guard.stop_heartbeat_writer();
        fs::write(&heartbeat_path, serde_json::to_vec_pretty(&stolen).unwrap()).unwrap();
        drop(guard);
        assert!(
            heartbeat_path.is_file(),
            "Drop must not delete a heartbeat it does not own"
        );
        let _ = fs::remove_dir_all(&home);
    }
}
