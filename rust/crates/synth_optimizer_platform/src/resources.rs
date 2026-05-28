use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceLeaseRecord {
    pub schema_version: String,
    pub resource_lease_id: String,
    pub lease_id: String,
    pub resource_kind: String,
    pub resource_id: String,
    pub subject_type: String,
    pub subject_id: String,
    pub run_id: String,
    pub status: String,
    #[serde(default)]
    pub lease_expires_at: Option<String>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

pub struct ResourceLeaseRecordInput<'a> {
    pub lease_id: &'a str,
    pub resource_kind: &'a str,
    pub resource_id: &'a str,
    pub run_id: &'a str,
    pub request_id: &'a str,
    pub status: &'a str,
    pub lease_expires_at: Option<String>,
    pub metadata: Map<String, Value>,
}

impl ResourceLeaseRecord {
    pub fn run_request(input: ResourceLeaseRecordInput<'_>) -> Self {
        let resource_kind = input.resource_kind.to_string();
        let resource_id = input.resource_id.to_string();
        Self {
            schema_version: "resource_lease_record.v1".to_string(),
            resource_lease_id: stable_id(
                "reslease",
                &[input.lease_id, &resource_kind, &resource_id],
            ),
            lease_id: input.lease_id.to_string(),
            resource_kind,
            resource_id,
            subject_type: "run_request".to_string(),
            subject_id: input.request_id.to_string(),
            run_id: input.run_id.to_string(),
            status: input.status.to_string(),
            lease_expires_at: input.lease_expires_at,
            metadata: input.metadata,
        }
    }
}

fn stable_id(prefix: &str, parts: &[&str]) -> String {
    let mut digest = Sha256::new();
    digest.update(prefix.as_bytes());
    for part in parts {
        digest.update(b"\0");
        digest.update(part.as_bytes());
    }
    let hex = format!("{:x}", digest.finalize());
    format!("{prefix}_{}", &hex[..16])
}
