use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperationRecord {
    pub schema_version: String,
    pub operation_id: String,
    pub operation_type: String,
    pub subject_type: String,
    pub subject_id: String,
    pub run_id: String,
    #[serde(default)]
    pub idempotency_key: Option<String>,
    pub status: String,
    #[serde(default)]
    pub attempt: u64,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

impl OperationRecord {
    pub fn run_request(
        operation_type: impl Into<String>,
        run_id: &str,
        request_id: &str,
        status: impl Into<String>,
        idempotency_key: Option<String>,
        metadata: Map<String, Value>,
    ) -> Self {
        let operation_type = operation_type.into();
        Self {
            schema_version: "operation_record.v1".to_string(),
            operation_id: stable_id("operation", &[&operation_type, request_id]),
            operation_type,
            subject_type: "run_request".to_string(),
            subject_id: request_id.to_string(),
            run_id: run_id.to_string(),
            idempotency_key,
            status: status.into(),
            attempt: 1,
            metadata,
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
