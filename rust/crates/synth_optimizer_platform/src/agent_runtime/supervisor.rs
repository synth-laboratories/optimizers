use serde::{Deserialize, Serialize};

/// Operator-visible launch/cleanup metadata for an agent runtime process.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SupervisorReceipt {
    pub substrate: String,
    pub process_id: Option<u32>,
    pub container_name: Option<String>,
    pub image: Option<String>,
    pub staging_dir: Option<String>,
    pub workspace_mount_path: Option<String>,
    pub cleanup_status: String,
}
