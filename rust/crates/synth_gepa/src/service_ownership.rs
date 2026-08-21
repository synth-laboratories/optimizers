use std::fs::{self, File, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
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
const DEFAULT_HEARTBEAT_STALE: Duration = Duration::from_secs(10);
const LOCK_RETRY: Duration = Duration::from_millis(50);
const LOCK_RETRY_BUDGET: Duration = Duration::from_secs(2);

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
                if let Some(peer) = read_heartbeat(&heartbeat_path)? {
                    let peer_pid = heartbeat_pid(&peer);
                    if peer_pid != std::process::id() && pid_is_alive(peer_pid) {
                        kill_pid(peer_pid);
                        adopted_crash = Some(orphan_sidecar_crash(&peer, peer_pid));
                    } else if peer_pid != std::process::id() {
                        adopted_crash = Some(stale_heartbeat_crash(&peer, peer_pid));
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
                        let peer_pid = heartbeat_pid(&peer);
                        if peer_pid != std::process::id() && pid_is_alive(peer_pid) {
                            kill_pid(peer_pid);
                            adopted_crash = Some(orphan_sidecar_crash(&peer, peer_pid));
                        } else if adopted_crash.is_none() {
                            adopted_crash = Some(stale_heartbeat_crash(&peer, peer_pid));
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

pub fn heartbeat_stale_after() -> Duration {
    std::env::var("SYNTH_GEPA_HEARTBEAT_STALE_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_HEARTBEAT_STALE)
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

pub fn owned_heartbeat_payload(
    config: &GepaServiceConfig,
    service_url: &str,
    started_at: &str,
    last_seen: Option<String>,
) -> Value {
    let mut payload = json!({
        "kind": "gepa-service",
        "schema": "synth.gepa_service.whoami.v1",
        "version": env!("CARGO_PKG_VERSION"),
        "source_id": service_id_for(config),
        "service_url": service_url,
        "bind": config.bind_addr.clone(),
        "pid": std::process::id(),
        "db_path": absolute_path(&config.db_path).display().to_string(),
        "workshop_instance_id": config.workshop_instance_id.clone().unwrap_or_default(),
        "worker_id": config.worker_id.clone(),
        "workers": config.worker_count,
        "lease_seconds": config.lease_seconds,
        "started_at": started_at,
        "run_roots": [],
    });
    if let Some(last_seen) = last_seen {
        if let Some(object) = payload.as_object_mut() {
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

fn peer_is_healthy(payload: &Value) -> bool {
    let pid = heartbeat_pid(payload);
    if pid == 0 || !pid_is_alive(pid) {
        return false;
    }
    let Some(last_seen) = payload.get("last_seen").and_then(Value::as_str) else {
        return false;
    };
    let Ok(timestamp) = OffsetDateTime::parse(last_seen, &Rfc3339) else {
        return false;
    };
    let age = OffsetDateTime::now_utc() - timestamp;
    age < heartbeat_stale_after()
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

fn stale_heartbeat_crash(peer: &Value, pid: u32) -> Value {
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

#[cfg(unix)]
fn pid_is_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    let rc = unsafe { libc::kill(pid as i32, 0) };
    if rc == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
fn pid_is_alive(pid: u32) -> bool {
    pid != 0 && pid == std::process::id()
}

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
            }
            other => panic!("expected AlreadyRunning, got {other:?}"),
        }
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn stale_heartbeat_is_adopted() {
        let home = scratch_home("stale");
        let db = home.join("workspace.sqlite");
        fs::write(&db, b"").unwrap();
        let config = test_config(db, "workshop-stale");
        let service_id = service_id_for(&config);
        let services = home.join("services");
        fs::create_dir_all(&services).unwrap();
        let heartbeat_path = services.join(format!("{service_id}.json"));
        let mut payload = owned_heartbeat_payload(
            &config,
            "http://127.0.0.1:9",
            "2000-01-01T00:00:00Z",
            Some("2000-01-01T00:00:00Z".to_string()),
        );
        payload["pid"] = json!(i32::MAX as u32);
        fs::write(
            &heartbeat_path,
            serde_json::to_vec_pretty(&payload).unwrap(),
        )
        .unwrap();

        let guard = acquire_service_ownership_in(home.clone(), &config)
            .expect("stale heartbeat must be adoptable");
        assert!(guard.lock_path().exists());
        let crash = guard
            .adopted_crash
            .as_ref()
            .expect("adopted crash record");
        assert_eq!(crash["cause"], "service_crash");
        assert_eq!(crash["reason"], "stale_heartbeat");
        assert_eq!(crash["pid"], i32::MAX as u32);
        guard.stop_heartbeat_writer();
        drop(guard);
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
