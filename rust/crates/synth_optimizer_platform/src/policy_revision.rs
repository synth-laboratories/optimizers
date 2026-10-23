//! Capability-negotiated publication shared by GEPA and its prompt consumers.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::error::{OptimizerError, Result};

#[derive(Default)]
pub(crate) struct RevisionTransport {
    capability: Option<Value>,
    prepared: BTreeMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn transport() -> RevisionTransport {
        let mut transport = RevisionTransport::default();
        transport.observe(&json!({"policy_revision": {
            "schema_version": "synth.policy-revision.v1",
            "profiles": ["next_rollout_update.v1"],
            "adapters": {"prompt_program": ["prompt_fields", "config"]}
        }}));
        transport
    }

    #[test]
    fn publish_prepare_pin_and_cache_without_inline_policy() {
        let calls = RefCell::new(Vec::new());
        let revision = format!("polrev_{}", "a".repeat(64));
        let post = |path: &str, body: &Value| {
            calls.borrow_mut().push((path.to_string(), body.clone()));
            Ok(if path == "/policy-revisions" {
                json!({"policy_revision_id": revision})
            } else {
                json!({"state": "ready", "actual_binding": {"policy_revision_id": revision}})
            })
        };
        let request = json!({"task": {"id": "row"}, "candidate": {"system": "A"},
            "candidate_overlay": {"candidate": {"system": "A"}},
            "policy": {"provider": "fixture", "model": "fixture"}});
        let mut transport = transport();
        for _ in 0..2 {
            let (pinned, id) = transport.pin(&request, post).unwrap();
            assert_eq!(id.as_deref(), Some(revision.as_str()));
            assert!(pinned.get("candidate").is_none());
            assert!(pinned.get("candidate_overlay").is_none());
            assert!(pinned.get("policy").is_none());
            assert_eq!(pinned["task"], request["task"]);
        }
        assert_eq!(calls.borrow().len(), 2);
        assert_eq!(
            calls.borrow()[0].1["components"]["prompt_fields"],
            request["candidate"]
        );
    }

    #[test]
    fn failed_or_substituted_preparation_never_pins() {
        let revision = format!("polrev_{}", "b".repeat(64));
        for state in ["failed", "ready"] {
            let result = transport().pin(&json!({"candidate": {"system": "A"}}), |path, _| {
                Ok(if path == "/policy-revisions" {
                    json!({"policy_revision_id": revision})
                } else {
                    json!({"state": state, "actual_binding": {"policy_revision_id": "wrong"}})
                })
            });
            assert!(result.is_err());
        }
    }

    #[test]
    fn legacy_target_keeps_its_versioned_wire_shape() {
        let request = json!({"candidate": {"system": "legacy"}});
        assert_eq!(
            RevisionTransport::default()
                .pin(&request, |_, _| panic!("no publish"))
                .unwrap(),
            (request, None)
        );
    }
}

impl RevisionTransport {
    pub(crate) fn observe(&mut self, metadata: &Value) {
        self.capability = metadata.get("policy_revision").cloned();
    }

    pub(crate) fn pin(
        &mut self,
        request: &Value,
        post: impl Fn(&str, &Value) -> Result<Value>,
    ) -> Result<(Value, Option<String>)> {
        let Some(capability) = &self.capability else {
            return Ok((request.clone(), None));
        };
        if capability.get("schema_version").and_then(Value::as_str)
            != Some("synth.policy-revision.v1")
        {
            return Err(OptimizerError::Container(
                "unsupported policy revision schema".into(),
            ));
        }
        let supported = capability
            .get("profiles")
            .and_then(Value::as_array)
            .is_some_and(|profiles| profiles.iter().any(|v| v == "next_rollout_update.v1"))
            && capability.pointer("/adapters/prompt_program").is_some();
        if !supported {
            return Ok((request.clone(), None));
        }
        if request
            .get("policy_revision_id")
            .and_then(Value::as_str)
            .is_some()
        {
            if request.get("candidate").is_some() || request.get("candidate_overlay").is_some() {
                return Err(OptimizerError::Container(
                    "policy pin contains inline candidate".into(),
                ));
            }
            return Ok((
                request.clone(),
                request
                    .get("policy_revision_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            ));
        }
        let candidate = request
            .get("candidate")
            .or_else(|| request.pointer("/candidate_overlay/candidate"))
            .ok_or_else(|| OptimizerError::Container("policy candidate missing".into()))?;
        if let Some(overlay) = request.pointer("/candidate_overlay/candidate") {
            if overlay != candidate {
                return Err(OptimizerError::Container(
                    "candidate overlay conflicts".into(),
                ));
            }
        }
        let manifest = json!({
            "schema_version": "synth.policy-revision.v1",
            "namespace": "optimizers", "name": "prompt", "harness": "prompt_program",
            "components": {"prompt_fields": candidate,
                "config": {"policy": request.get("policy").cloned().unwrap_or(Value::Null)}}
        });
        let key = serde_json::to_string(&manifest)?;
        let revision = if let Some(revision) = self.prepared.get(&key) {
            revision.clone()
        } else {
            let published = post("/policy-revisions", &manifest)?;
            let revision = published
                .get("policy_revision_id")
                .and_then(Value::as_str)
                .filter(|value| {
                    value.starts_with("polrev_")
                        && value.len() == 71
                        && value[7..]
                            .bytes()
                            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
                })
                .ok_or_else(|| OptimizerError::Container("invalid policy revision receipt".into()))?
                .to_owned();
            let ready = post(
                &format!("/policy-revisions/{revision}/prepare"),
                &json!({
                    "operation_id": format!("optimizers-prepare-{revision}"),
                    "profile": "next_rollout_update.v1", "timeout_s": 10,
                }),
            )?;
            if ready.get("state").and_then(Value::as_str) != Some("ready")
                || ready
                    .pointer("/actual_binding/policy_revision_id")
                    .and_then(Value::as_str)
                    != Some(revision.as_str())
            {
                return Err(OptimizerError::Container(
                    "policy preparation did not confirm binding".into(),
                ));
            }
            self.prepared.insert(key, revision.clone());
            revision
        };
        let mut pinned = request.clone();
        let object = pinned
            .as_object_mut()
            .ok_or_else(|| OptimizerError::Container("rollout must be an object".into()))?;
        object.remove("candidate");
        object.remove("candidate_overlay");
        object.remove("policy");
        object.insert("policy_revision_id".into(), json!(revision));
        Ok((pinned, Some(revision)))
    }
}
