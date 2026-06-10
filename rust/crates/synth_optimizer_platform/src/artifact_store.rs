use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::error::{OptimizerError, Result};

pub trait RunArtifactStore: std::fmt::Debug + Send + Sync {
    fn write_json(&self, path_key: &str, value: &Value) -> Result<StoredRunArtifact>;
    fn write_bytes(&self, path_key: &str, bytes: &[u8]) -> Result<StoredRunArtifact>;
    fn open_sqlite_staging(&self, name: &str) -> Result<PathBuf>;
}

#[derive(Clone, Debug)]
pub struct LocalDevStore {
    root_dir: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredRunArtifact {
    pub path_key: String,
    pub uri: String,
    pub local_path: Option<String>,
    pub bytes: u64,
    pub sha256: String,
}

impl LocalDevStore {
    pub fn new(root_dir: impl Into<PathBuf>) -> Self {
        Self {
            root_dir: root_dir.into(),
        }
    }

    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    pub fn resolve_path_key(&self, path_key: &str) -> Result<PathBuf> {
        let key = validate_path_key(path_key)?;
        Ok(self.root_dir.join(key))
    }

    pub fn key_for_path(&self, path: &Path) -> Option<String> {
        let relative = path.strip_prefix(&self.root_dir).ok()?;
        let key = relative
            .components()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        if key.is_empty() {
            None
        } else {
            Some(key)
        }
    }
}

impl RunArtifactStore for LocalDevStore {
    fn write_json(&self, path_key: &str, value: &Value) -> Result<StoredRunArtifact> {
        let text = serde_json::to_string_pretty(value)?;
        self.write_bytes(path_key, format!("{text}\n").as_bytes())
    }

    fn write_bytes(&self, path_key: &str, bytes: &[u8]) -> Result<StoredRunArtifact> {
        let path = self.resolve_path_key(path_key)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| OptimizerError::io(parent, source))?;
        }
        fs::write(&path, bytes).map_err(|source| OptimizerError::io(&path, source))?;
        Ok(stored_artifact(path_key, &path, bytes))
    }

    fn open_sqlite_staging(&self, name: &str) -> Result<PathBuf> {
        let path = self.resolve_path_key(name)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| OptimizerError::io(parent, source))?;
        }
        Ok(path)
    }
}

fn validate_path_key(path_key: &str) -> Result<&str> {
    let key = path_key.trim().trim_start_matches('/');
    if key.is_empty() {
        return Err(OptimizerError::Config(
            "run artifact path_key must not be empty".to_string(),
        ));
    }
    let path = Path::new(key);
    if path.is_absolute() {
        return Err(OptimizerError::Config(format!(
            "run artifact path_key must be relative: {path_key}"
        )));
    }
    for component in path.components() {
        if !matches!(component, Component::Normal(_)) {
            return Err(OptimizerError::Config(format!(
                "run artifact path_key contains invalid component: {path_key}"
            )));
        }
    }
    Ok(key)
}

fn stored_artifact(path_key: &str, path: &Path, bytes: &[u8]) -> StoredRunArtifact {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    StoredRunArtifact {
        path_key: path_key.trim().trim_start_matches('/').to_string(),
        uri: format!("file://{}", path.display()),
        local_path: Some(path.display().to_string()),
        bytes: bytes.len() as u64,
        sha256: format!("{:x}", hasher.finalize()),
    }
}
