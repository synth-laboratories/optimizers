//! `synth.correlation.v1`: the envelope an experiment stamps on every trial.
//!
//! An optimizer run that an experiment dispatched must be findable from the
//! experiment side, and the arm a rollout belongs to must be readable from the
//! rollout side. One small record carried unchanged through the run request,
//! the resolved config, the manifest, the registry, and the rollout metadata is
//! what makes both directions work without any of those growing a private
//! notion of what an arm is.
//!
//! Two deliberate omissions.
//!
//! There is no run id. The service mints its own and reports it back; a caller
//! supplying one would be asserting authority over a namespace it does not own,
//! and resume would then have two candidate truths about which run a trial is.
//!
//! There is no digest. The producer computes one from its own canonical
//! encoding and compares it against the envelope it gets back; recomputing it
//! here would mean two languages agreeing byte for byte on JSON
//! canonicalisation forever, which is a promise not worth making for a field
//! nothing on this side reads.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{OptimizerError, Result};

pub const CORRELATION_SCHEMA_VERSION: &str = "synth.correlation.v1";

/// The only keys an index is asked to carry.
///
/// Indexing every declared factor would make a trace index a second, divergent
/// copy of the plan. These three are enough to get from a trace back to the
/// sealed envelope, which is the authority.
pub const ALIAS_KEYS: [&str; 3] = ["experiment_id", "trial_id", "candidate_id"];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubjectRef {
    pub subject_kind: String,
    pub subject_id: String,
    pub subject_content_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_subject_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorrelationEnvelope {
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    pub experiment_id: String,
    pub arm_id: String,
    pub block_id: String,
    pub replicate: u32,
    pub trial_id: String,
    pub plan_digest: String,
    pub subject: SubjectRef,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_id: Option<String>,
}

fn default_schema_version() -> String {
    CORRELATION_SCHEMA_VERSION.to_string()
}

impl CorrelationEnvelope {
    /// Refuse a malformed envelope at admission, while no run record exists.
    ///
    /// A half-formed envelope is worse than none: it would be persisted into
    /// manifests and traces and only fail to join much later, when the run it
    /// was supposed to identify has already cost something.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != CORRELATION_SCHEMA_VERSION {
            return Err(OptimizerError::Config(format!(
                "correlation.schema_version must be {CORRELATION_SCHEMA_VERSION}, got {:?}",
                self.schema_version
            )));
        }
        for (field, value) in [
            ("experiment_id", &self.experiment_id),
            ("arm_id", &self.arm_id),
            ("block_id", &self.block_id),
            ("trial_id", &self.trial_id),
            ("subject.subject_kind", &self.subject.subject_kind),
            ("subject.subject_id", &self.subject.subject_id),
        ] {
            if value.trim().is_empty() {
                return Err(OptimizerError::Config(format!(
                    "correlation.{field} must be a non-empty string"
                )));
            }
        }
        require_sha256("correlation.plan_digest", &self.plan_digest)?;
        require_sha256(
            "correlation.subject.subject_content_digest",
            &self.subject.subject_content_digest,
        )?;
        Ok(())
    }

    /// The bounded subset a downstream index may carry.
    pub fn aliases(&self) -> BTreeMap<String, String> {
        let mut aliases = BTreeMap::new();
        aliases.insert("experiment_id".to_string(), self.experiment_id.clone());
        aliases.insert("trial_id".to_string(), self.trial_id.clone());
        if let Some(candidate_id) = self.candidate_id.as_ref() {
            aliases.insert("candidate_id".to_string(), candidate_id.clone());
        }
        aliases
    }
}

fn require_sha256(field: &str, value: &str) -> Result<()> {
    let valid = value
        .strip_prefix("sha256:")
        .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit()));
    if valid {
        Ok(())
    } else {
        Err(OptimizerError::Config(format!(
            "{field} must be a sha256:<64 hex> digest, got {value:?}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn envelope() -> CorrelationEnvelope {
        serde_json::from_value(json!({
            "schema_version": CORRELATION_SCHEMA_VERSION,
            "experiment_id": "luna-effort-v1",
            "arm_id": "arm_6edf53cf5835",
            "block_id": "seed:104",
            "replicate": 0,
            "trial_id": "t676b09f94e51e2f3",
            "plan_digest": format!("sha256:{}", "ab".repeat(32)),
            "subject": {
                "subject_kind": "proposer-policy",
                "subject_id": "gpt-5.6-luna@low",
                "subject_content_digest": format!("sha256:{}", "cd".repeat(32)),
            },
            "candidate_id": "cand_7",
        }))
        .expect("envelope parses")
    }

    #[test]
    fn a_well_formed_envelope_validates_and_round_trips() {
        let parsed = envelope();
        parsed.validate().expect("valid");
        let again: CorrelationEnvelope =
            serde_json::from_value(serde_json::to_value(&parsed).unwrap()).unwrap();
        assert_eq!(parsed, again);
    }

    #[test]
    fn aliases_are_bounded_to_three_keys() {
        let aliases = envelope().aliases();
        assert_eq!(
            aliases.keys().cloned().collect::<Vec<_>>(),
            vec!["candidate_id", "experiment_id", "trial_id"]
        );
        for key in aliases.keys() {
            assert!(ALIAS_KEYS.contains(&key.as_str()));
        }
    }

    #[test]
    fn a_missing_candidate_id_drops_the_alias_rather_than_emitting_an_empty_one() {
        let mut parsed = envelope();
        parsed.candidate_id = None;
        assert_eq!(parsed.aliases().len(), 2);
        let encoded = serde_json::to_value(&parsed).unwrap();
        assert!(encoded.get("candidate_id").is_none());
    }

    #[test]
    fn an_unknown_field_is_refused_rather_than_silently_dropped() {
        let mut value = serde_json::to_value(envelope()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("run_id".into(), json!("gepa_1234"));
        // A caller cannot smuggle authority over the run namespace through here.
        assert!(serde_json::from_value::<CorrelationEnvelope>(value).is_err());
    }

    #[test]
    fn a_non_digest_plan_reference_is_refused_at_admission() {
        let mut parsed = envelope();
        parsed.plan_digest = "plan-3".to_string();
        let error = parsed.validate().expect_err("must refuse");
        assert!(error.to_string().contains("plan_digest"));
    }

    #[test]
    fn a_blank_identifier_is_refused() {
        let mut parsed = envelope();
        parsed.arm_id = "  ".to_string();
        assert!(parsed.validate().is_err());
    }

    /// The exact bytes `synth_optimizers.experiment` emits for one envelope.
    ///
    /// Pinned on both sides. `deny_unknown_fields` means a field added in
    /// Python and not here would be refused at admission rather than dropped,
    /// and this is where that shows up as a test failure instead of a rejected
    /// run in the field.
    const PYTHON_WIRE_FORM: &str = r#"{"arm_id":"arm_6edf53cf5835","block_id":"seed:104","experiment_id":"luna-effort-v1","plan_digest":"sha256:abababababababababababababababababababababababababababababababab","replicate":0,"schema_version":"synth.correlation.v1","subject":{"subject_content_digest":"sha256:cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd","subject_id":"gpt-5.6-luna@low","subject_kind":"proposer-policy"},"trial_id":"t676b09f94e51e2f3"}"#;

    #[test]
    fn the_python_wire_form_deserializes_here_unchanged() {
        let parsed: CorrelationEnvelope =
            serde_json::from_str(PYTHON_WIRE_FORM).expect("python envelope parses");
        parsed.validate().expect("valid");
        assert_eq!(parsed.trial_id, "t676b09f94e51e2f3");
        assert_eq!(parsed.subject.subject_id, "gpt-5.6-luna@low");
        assert!(parsed.candidate_id.is_none());
        assert!(parsed.subject.parent_subject_id.is_none());

        // And back out: an absent optional stays absent, so the record survives
        // a format with no null (TOML) without changing its digest.
        let encoded = serde_json::to_value(&parsed).unwrap();
        assert!(encoded.get("candidate_id").is_none());
        assert!(encoded["subject"].get("parent_subject_id").is_none());
    }

    #[test]
    fn a_foreign_schema_version_is_refused() {
        let mut parsed = envelope();
        parsed.schema_version = "synth.correlation.v2".to_string();
        assert!(parsed.validate().is_err());
    }
}
