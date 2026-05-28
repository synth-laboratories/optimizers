//! Soft and hard storage limits for the optimizer runs directory.
//!
//! GEPA writes per-run artifacts (events.jsonl, candidate registries,
//! rollout traces) under `output_dir/<run_id>/`. Local dev hit ENOSPC
//! after ~40 accumulated runs on 2026-05-22 — the in-flight run's jsonl
//! got corrupted mid-write, and every subsequent launch failed at startup
//! when its first artifact write succeeded but its second did not. This
//! module is the structural defense:
//!
//! - **Soft limit** — evaluated once at the top of a run launch.
//!   Aggregate `output_dir` usage at or above the soft floor rejects
//!   the new run with [`OptimizerError::DiskBudgetExceeded`] carrying
//!   `limit_kind="soft"`. Existing runs are not interrupted.
//! - **Hard limit** — evaluated before each artifact / event write.
//!   Aggregate usage at or above the hard floor refuses the write and
//!   returns [`OptimizerError::DiskBudgetExceeded`] with
//!   `limit_kind="hard"`. The caller can finalize the run cleanly
//!   instead of partial-writing a corrupted file.
//!
//! Both limits are enforced against the **same** monitored directory —
//! the aggregate `output_dir` parent of every run, so the budget caps
//! cross-run growth, not per-run.
//!
//! ### Caching
//!
//! Walking `output_dir` is O(files-on-disk). For a runs directory with
//! 40+ runs and thousands of files each that is multi-millisecond. The
//! hard-limit gate fires on every event write (potentially many per
//! second), so a full walk per call would dominate runtime. We cache the
//! most recent walk for [`CACHE_TTL`] and let callers
//! [`DiskBudget::note_appended_bytes`] to keep the cached counter
//! roughly accurate between full walks. After a soft TTL window expires
//! the next call recomputes.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{OptimizerError, Result};

const DEFAULT_SOFT_LIMIT_GB: f64 = 5.0;
const DEFAULT_HARD_LIMIT_GB: f64 = 10.0;
const CACHE_TTL: Duration = Duration::from_secs(10);
const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Tunable storage limits for the optimizer runs directory.
///
/// Defaults are conservative for local development (5 GiB soft, 10 GiB
/// hard). Production deployments should override with the largest values
/// the host can sustain — the budget is meant to prevent ENOSPC, not to
/// be the day-to-day capacity plan.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiskBudgetConfig {
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default = "default_soft_limit_gb")]
    pub soft_limit_gb: f64,
    #[serde(default = "default_hard_limit_gb")]
    pub hard_limit_gb: f64,
    /// Override the monitored directory. Defaults to `run.output_dir`.
    #[serde(default)]
    pub path: Option<PathBuf>,
}

fn default_enabled() -> bool {
    true
}

fn default_soft_limit_gb() -> f64 {
    DEFAULT_SOFT_LIMIT_GB
}

fn default_hard_limit_gb() -> f64 {
    DEFAULT_HARD_LIMIT_GB
}

impl Default for DiskBudgetConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            soft_limit_gb: default_soft_limit_gb(),
            hard_limit_gb: default_hard_limit_gb(),
            path: None,
        }
    }
}

impl DiskBudgetConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        if !(self.soft_limit_gb > 0.0) {
            return Err(OptimizerError::Config(format!(
                "disk_budget.soft_limit_gb must be > 0, got {}",
                self.soft_limit_gb
            )));
        }
        if !(self.hard_limit_gb > 0.0) {
            return Err(OptimizerError::Config(format!(
                "disk_budget.hard_limit_gb must be > 0, got {}",
                self.hard_limit_gb
            )));
        }
        if self.hard_limit_gb < self.soft_limit_gb {
            return Err(OptimizerError::Config(format!(
                "disk_budget.hard_limit_gb ({}) must be >= soft_limit_gb ({})",
                self.hard_limit_gb, self.soft_limit_gb
            )));
        }
        Ok(())
    }

    pub fn soft_limit_bytes(&self) -> u64 {
        gib_to_bytes(self.soft_limit_gb)
    }

    pub fn hard_limit_bytes(&self) -> u64 {
        gib_to_bytes(self.hard_limit_gb)
    }
}

fn gib_to_bytes(gib: f64) -> u64 {
    (gib.max(0.0) * BYTES_PER_GIB) as u64
}

/// Result of one budget evaluation. Carries the observed usage so
/// callers can surface it in logs or events without re-walking.
#[derive(Clone, Copy, Debug)]
pub enum DiskBudgetState {
    Healthy { used_bytes: u64 },
    SoftExceeded { used_bytes: u64, limit_bytes: u64 },
    HardExceeded { used_bytes: u64, limit_bytes: u64 },
}

impl DiskBudgetState {
    pub fn used_bytes(&self) -> u64 {
        match self {
            Self::Healthy { used_bytes }
            | Self::SoftExceeded { used_bytes, .. }
            | Self::HardExceeded { used_bytes, .. } => *used_bytes,
        }
    }

    pub fn is_hard(&self) -> bool {
        matches!(self, Self::HardExceeded { .. })
    }

    pub fn is_soft_or_worse(&self) -> bool {
        matches!(self, Self::SoftExceeded { .. } | Self::HardExceeded { .. })
    }
}

/// Cheap-to-clone handle holding the configured limits, the monitored
/// path, and a cached size to amortize directory walks across many
/// calls.
#[derive(Clone)]
pub struct DiskBudget {
    inner: Arc<DiskBudgetInner>,
}

struct DiskBudgetInner {
    config: DiskBudgetConfig,
    monitored_path: PathBuf,
    cache: Mutex<DiskBudgetCache>,
}

#[derive(Debug, Default)]
struct DiskBudgetCache {
    last_check: Option<Instant>,
    last_size_bytes: u64,
}

impl DiskBudget {
    pub fn new(config: DiskBudgetConfig, output_dir: impl Into<PathBuf>) -> Result<Self> {
        config.validate()?;
        let monitored_path = config.path.clone().unwrap_or_else(|| output_dir.into());
        Ok(Self {
            inner: Arc::new(DiskBudgetInner {
                config,
                monitored_path,
                cache: Mutex::new(DiskBudgetCache::default()),
            }),
        })
    }

    pub fn enabled(&self) -> bool {
        self.inner.config.enabled
    }

    pub fn monitored_path(&self) -> &Path {
        &self.inner.monitored_path
    }

    pub fn config(&self) -> &DiskBudgetConfig {
        &self.inner.config
    }

    /// Re-read the monitored directory (subject to the cache TTL) and
    /// classify the result.
    pub fn evaluate(&self) -> Result<DiskBudgetState> {
        let used = self.current_used_bytes()?;
        Ok(self.classify(used))
    }

    /// Reject the launch of a new run when usage is at or above the
    /// soft limit. Healthy and `HardExceeded` states are not callers
    /// of this — `HardExceeded` implies soft-exceeded too.
    pub fn require_below_soft(&self) -> Result<()> {
        if !self.enabled() {
            return Ok(());
        }
        match self.evaluate()? {
            DiskBudgetState::Healthy { .. } => Ok(()),
            DiskBudgetState::SoftExceeded {
                used_bytes,
                limit_bytes,
            } => Err(self.error(used_bytes, limit_bytes, "soft")),
            DiskBudgetState::HardExceeded {
                used_bytes,
                limit_bytes,
            } => {
                // Hard implies soft; report as soft so the operator sees
                // the launch-time gate, not the write-time gate.
                Err(self.error(
                    used_bytes,
                    self.inner.config.soft_limit_bytes().max(limit_bytes),
                    "soft",
                ))
            }
        }
    }

    /// Reject the next artifact write when usage is at or above the
    /// hard limit. Soft-exceeded but not hard-exceeded continues writing
    /// — the soft limit is only a launch-time gate.
    pub fn require_below_hard(&self) -> Result<()> {
        if !self.enabled() {
            return Ok(());
        }
        match self.evaluate()? {
            DiskBudgetState::Healthy { .. } | DiskBudgetState::SoftExceeded { .. } => Ok(()),
            DiskBudgetState::HardExceeded {
                used_bytes,
                limit_bytes,
            } => Err(self.error(used_bytes, limit_bytes, "hard")),
        }
    }

    /// Account for bytes appended outside a full walk so the cached
    /// counter stays roughly correct between walks. Saturates rather
    /// than overflowing — the next full walk will correct any drift.
    pub fn note_appended_bytes(&self, bytes: u64) {
        if !self.enabled() {
            return;
        }
        let mut guard = self.inner.cache.lock().expect("disk_budget cache poisoned");
        guard.last_size_bytes = guard.last_size_bytes.saturating_add(bytes);
    }

    fn current_used_bytes(&self) -> Result<u64> {
        {
            let guard = self.inner.cache.lock().expect("disk_budget cache poisoned");
            if let Some(last) = guard.last_check {
                if last.elapsed() < CACHE_TTL {
                    return Ok(guard.last_size_bytes);
                }
            }
        }
        let used = directory_size_bytes(&self.inner.monitored_path)?;
        let mut guard = self.inner.cache.lock().expect("disk_budget cache poisoned");
        guard.last_check = Some(Instant::now());
        guard.last_size_bytes = used;
        Ok(used)
    }

    fn classify(&self, used_bytes: u64) -> DiskBudgetState {
        let hard = self.inner.config.hard_limit_bytes();
        let soft = self.inner.config.soft_limit_bytes();
        if used_bytes >= hard {
            DiskBudgetState::HardExceeded {
                used_bytes,
                limit_bytes: hard,
            }
        } else if used_bytes >= soft {
            DiskBudgetState::SoftExceeded {
                used_bytes,
                limit_bytes: soft,
            }
        } else {
            DiskBudgetState::Healthy { used_bytes }
        }
    }

    fn error(&self, used_bytes: u64, limit_bytes: u64, limit_kind: &'static str) -> OptimizerError {
        OptimizerError::DiskBudgetExceeded {
            path: self.inner.monitored_path.clone(),
            used_bytes,
            limit_bytes,
            limit_kind,
        }
    }
}

/// Recursive byte-count of every regular file under `path`. Symlinks
/// are not followed (matches `du -s`'s default behavior on macOS and
/// avoids double-counting when runs share workspace inputs via
/// symlink).
pub fn directory_size_bytes(path: &Path) -> Result<u64> {
    if !path.exists() {
        return Ok(0);
    }
    let mut total: u64 = 0;
    let mut stack: Vec<PathBuf> = vec![path.to_path_buf()];
    while let Some(current) = stack.pop() {
        let read_dir = match std::fs::read_dir(&current) {
            Ok(rd) => rd,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => continue,
            Err(source) => return Err(OptimizerError::io(&current, source)),
        };
        for entry in read_dir {
            let entry = entry.map_err(|source| OptimizerError::io(&current, source))?;
            let metadata = match entry.metadata() {
                Ok(m) => m,
                Err(source) if source.kind() == std::io::ErrorKind::NotFound => continue,
                Err(source) => return Err(OptimizerError::io(entry.path(), source)),
            };
            let file_type = metadata.file_type();
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                stack.push(entry.path());
            } else if file_type.is_file() {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    fn write_file(path: &Path, bytes: usize) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut f = fs::File::create(path).unwrap();
        f.write_all(&vec![0u8; bytes]).unwrap();
    }

    #[test]
    fn validate_rejects_zero_or_inverted() {
        let mut cfg = DiskBudgetConfig::default();
        cfg.soft_limit_gb = 0.0;
        assert!(cfg.validate().is_err());
        let mut cfg = DiskBudgetConfig::default();
        cfg.hard_limit_gb = 1.0;
        cfg.soft_limit_gb = 2.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_allows_disabled_with_any_values() {
        let cfg = DiskBudgetConfig {
            enabled: false,
            soft_limit_gb: 0.0,
            hard_limit_gb: 0.0,
            path: None,
        };
        cfg.validate().unwrap();
    }

    #[test]
    fn directory_size_sums_regular_files() {
        let tmp = std::env::temp_dir().join(format!("disk_budget_test_{}", uuid::Uuid::new_v4()));
        write_file(&tmp.join("a/b/c.bin"), 100);
        write_file(&tmp.join("a/d.bin"), 250);
        write_file(&tmp.join("e.bin"), 50);
        let total = directory_size_bytes(&tmp).unwrap();
        assert_eq!(total, 400);
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn classifies_healthy_soft_hard_thresholds() {
        let tmp =
            std::env::temp_dir().join(format!("disk_budget_classify_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&tmp).unwrap();
        let cfg = DiskBudgetConfig {
            enabled: true,
            soft_limit_gb: 1.0 / 1024.0 / 1024.0, // 1 KiB
            hard_limit_gb: 2.0 / 1024.0 / 1024.0, // 2 KiB
            path: Some(tmp.clone()),
        };
        let budget = DiskBudget::new(cfg, &tmp).unwrap();
        // empty → healthy
        assert!(budget.require_below_soft().is_ok());
        // 1.5 KiB → soft exceeded but not hard
        write_file(&tmp.join("x.bin"), 1536);
        // force cache refresh
        budget.inner.cache.lock().unwrap().last_check = None;
        assert!(matches!(
            budget.evaluate().unwrap(),
            DiskBudgetState::SoftExceeded { .. }
        ));
        assert!(budget.require_below_soft().is_err());
        assert!(budget.require_below_hard().is_ok());
        // 3 KiB total → hard exceeded
        write_file(&tmp.join("y.bin"), 1600);
        budget.inner.cache.lock().unwrap().last_check = None;
        assert!(matches!(
            budget.evaluate().unwrap(),
            DiskBudgetState::HardExceeded { .. }
        ));
        assert!(budget.require_below_hard().is_err());
        fs::remove_dir_all(&tmp).ok();
    }
}
