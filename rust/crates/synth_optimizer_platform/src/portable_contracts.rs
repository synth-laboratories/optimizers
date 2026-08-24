//! v0.8 portable contract canonicalization.
//!
//! The JSON schemas and golden corpus in `contracts/synth-spine-v1` are the
//! authority. This module is a Rust consumer; it does not define another ID.

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use thiserror::Error;

const DOMAIN: &[u8] = b"synth.canonical-json.v1\0";
const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

#[derive(Debug, Error, PartialEq, Eq)]
#[error("{code}{detail}")]
pub struct PortableContractError {
    pub code: &'static str,
    detail: String,
}

impl PortableContractError {
    fn new(code: &'static str) -> Self {
        Self {
            code,
            detail: String::new(),
        }
    }
}

fn encode(value: &Value, output: &mut String) -> Result<(), PortableContractError> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(true) => output.push_str("true"),
        Value::Bool(false) => output.push_str("false"),
        Value::String(text) => {
            output.push_str(&serde_json::to_string(text).expect("string encoding"))
        }
        Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                if value.unsigned_abs() > MAX_SAFE_INTEGER {
                    return Err(PortableContractError::new("canonical_integer_range"));
                }
                output.push_str(&value.to_string());
            } else if let Some(value) = number.as_u64() {
                if value > MAX_SAFE_INTEGER {
                    return Err(PortableContractError::new("canonical_integer_range"));
                }
                output.push_str(&value.to_string());
            } else {
                return Err(PortableContractError::new("canonical_float_forbidden"));
            }
        }
        Value::Array(items) => {
            output.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                encode(item, output)?;
            }
            output.push(']');
        }
        Value::Object(object) => {
            let mut keys = object.keys().collect::<Vec<_>>();
            keys.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
            output.push('{');
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(&serde_json::to_string(key).expect("key encoding"));
                output.push(':');
                encode(&object[*key], output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

pub fn canonical_json(
    value: &Value,
    omit_self_digest: bool,
) -> Result<Vec<u8>, PortableContractError> {
    let owned;
    let value = if omit_self_digest {
        let object = value
            .as_object()
            .ok_or_else(|| PortableContractError::new("canonical_self_digest_root"))?;
        let mut copy: Map<String, Value> = object.clone();
        copy.remove("digest");
        owned = Value::Object(copy);
        &owned
    } else {
        value
    };
    let mut output = String::new();
    encode(value, &mut output)?;
    Ok(output.into_bytes())
}

pub fn canonical_digest(
    value: &Value,
    omit_self_digest: bool,
) -> Result<String, PortableContractError> {
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN);
    hasher.update(canonical_json(value, omit_self_digest)?);
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

pub fn verify_self_digest(value: &Value) -> Result<String, PortableContractError> {
    let offered = value
        .get("digest")
        .and_then(Value::as_str)
        .ok_or_else(|| PortableContractError::new("digest_malformed"))?;
    if offered.len() != 71
        || !offered.starts_with("sha256:")
        || !offered[7..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PortableContractError::new("digest_malformed"));
    }
    let computed = canonical_digest(value, true)?;
    if offered != computed {
        return Err(PortableContractError::new("digest_mismatch"));
    }
    Ok(computed)
}
