//! Typed shared contract for the container-first RL plane.
//!
//! This is the Rust half of the contract whose Python half lives in
//! `src/synth_optimizers/contracts/`. It exists for one reason: an executor
//! must be able to decide, before it spends anything, whether a container can
//! honor a particular run. Everything here is therefore fail-closed — a missing
//! route, an unparseable verdict, an absent clause answer, or a changed
//! capability document is an error, never a warning and never a default.
//!
//! Deliberate non-goals, mirrored from the Python side:
//!
//! - No type here names a task, a harness, an environment, or a model. The
//!   route table is declared by the container and read; it is never guessed and
//!   never dispatched on by name.
//! - Field names match the Python records field-for-field, because a batch is
//!   assembled from those records and the two planes must reconcile by mapping
//!   rather than by rewrite. Where the design note and the Python records
//!   disagree on a name, the Python name wins and a `serde(alias)` accepts the
//!   note's spelling; each such case is commented.
//!
//! Style parallels `container_contract.rs`: serde-defaulted fields so a partial
//! document still decodes into something a validator can reject by name,
//! `#[serde(flatten)] extra: JsonMap` so a container may add keys without
//! breaking decode, and `OptimizerError::Container` for every rejection.
//!
//! This root module owns the declared contract, the shared vocabulary, and the
//! capability content hash. The wire documents live in the submodules below and
//! are re-exported here, so `cispo_contract::X` is the only path callers need.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::cache::stable_json_hash;
use crate::container_contract::JsonMap;
use crate::error::{OptimizerError, Result};

pub mod attempts;
pub mod capabilities;
pub mod handshake;

pub use attempts::{
    decode_cispo_reward_receipt, decode_cispo_rollout_ack, decode_cispo_rollout_state,
    decode_cispo_trace_reference, CispoRewardReceipt, CispoRolloutAck, CispoRolloutState,
    CispoRolloutSubmission, CispoTraceReference, HorizonEvidenceDoc, RewardChannelDoc,
    RolloutInstanceState,
};
pub use capabilities::{
    decode_cispo_capabilities, AgentInstanceDoc, CispoCapabilityPreflight, CispoCapabilityResponse,
    CommunicationChannelDoc, EvidenceCapabilities, HorizonDoc, LifecycleCapabilities,
    RendererProfileDoc, RewardAuthorityCapabilities, TeamDoc, TopologyDoc,
};
pub use handshake::{
    decode_cispo_handshake_verdict, CispoHandshakeRequest, CispoHandshakeVerdict, ClauseVerdict,
    HandshakeClauseVerdict, HandshakeContainerClock, HandshakeExecutorClock, HandshakeObligations,
    HandshakeOptimizerIdentity, HandshakePolicyRequest, HandshakeRunPlanRequest,
    HandshakeTasksetRequest, HandshakeTopologyRequest, TasksetResolutionEntry,
};

/// The contract version this crate speaks. A container advertising anything
/// else is refused during preflight rather than probed for compatibility.
pub const CISPO_OPTIMIZER_CONTRACT_VERSION: &str = "synth_optimizers.cispo.v1";

/// Schema versions, mirrored from the Python contract modules so the two sides
/// cannot drift silently. `contracts/rl_records.py`, `rl_identity.py`, and
/// `rl_clauses.py` are the authorities; these constants must equal theirs.
pub const CISPO_CAPABILITIES_SCHEMA_VERSION: &str = "training.rollout.capabilities.v1";
pub const CISPO_HANDSHAKE_SCHEMA_VERSION: &str = "cispo.handshake.v1";
pub const CISPO_RENDERER_PROFILE_SCHEMA_VERSION: &str = "cispo.renderer_profile.v1";
pub const CISPO_TOPOLOGY_SCHEMA_VERSION: &str = "cispo.topology.v1";
pub const CISPO_TRAINABLE_EPISODE_SCHEMA_VERSION: &str = "cispo.trainable_episode.v1";
pub const CISPO_REWARD_RECORD_SCHEMA_VERSION: &str = "cispo.reward_record.v1";

/// Declared route keys the executor requires. A container may add or rename any
/// route, but it may not omit one of these and it may not expect the executor
/// to guess a path that is absent from `/metadata`.
pub const CISPO_MANDATORY_ROUTES: &[&str] = &[
    "health_route",
    "capabilities_route",
    "handshake_route",
    "taskset_route",
    "taskset_tasks_route",
    "policy_bind_route",
    "rollout_route",
    "rollout_state_route",
    "rollout_events_route",
    "rollout_renew_route",
    "rollout_terminate_route",
    "trace_route",
    "reward_route",
];

/// Declared route keys that may be absent. Each one is optional because the
/// capability it serves is itself optional in the contract, not because it is
/// nice-to-have:
///
/// - `topology_route`: the roster is already a field of the hashed capability
///   document, so a container with one fixed topology has nothing further to
///   serve. The route is only needed when rows in the same taskset resolve to
///   different rosters and each row must name its own `topology_ref`.
/// - `policy_set_bind_route`: an atomic all-instance binding only exists for a
///   joint episode. A single-instance run binds through `policy_bind_route`,
///   and the topology-scoped clauses do not apply to it at all.
/// - `rollout_finalize_route`: a separate finalize step is the staged
///   `running -> awaiting_score` path, which is the deferred-scoring
///   capability. A container that quiesces natively and returns its sealed
///   trace and materialized reward together is compliant without it; the
///   quiescence attestation then rides on the reward receipt's horizon
///   evidence.
/// - `artifacts_route`: `evidence.artifact_reference` is an optional clause and
///   durable artifact handoff is an optional capability. A container that
///   inlines its trace has no inventory to publish.
///
/// Everything else is mandatory. In particular `rollout_events_route` is *not*
/// optional: the optional capability is the streaming transport, while the
/// cursored GET that makes restart recovery cheap is required, and
/// `recovery.restart` is a mandatory clause.
pub const CISPO_OPTIONAL_ROUTES: &[&str] = &[
    "topology_route",
    "policy_set_bind_route",
    "rollout_finalize_route",
    "artifacts_route",
];

/// Clause registry, mirrored from `contracts/rl_clauses.py`. The list is
/// generic: no clause names a task, a harness, or an environment, and a
/// container may answer every clause without knowing which optimizer asked.
pub const CISPO_CLAUSE_GROUPS: &[(&str, &[&str])] = &[
    ("contract", &["contract.version", "contract.routes"]),
    (
        "discovery",
        &[
            "discovery.taskset",
            "discovery.task_digests",
            "discovery.topology",
        ],
    ),
    (
        "policy",
        &[
            "policy.binding_transport",
            "policy.renderer_profile_match",
            "policy.revision_immutability",
            "policy.no_embedded_credentials",
            "policy.session_scoped_origin",
        ],
    ),
    (
        "lifecycle",
        &[
            "lifecycle.idempotency",
            "lifecycle.lease_renewal",
            "lifecycle.cancellation",
            "lifecycle.concurrency",
            "lifecycle.exactly_one_terminal",
            "lifecycle.pause_resume",
        ],
    ),
    (
        "evidence",
        &[
            "evidence.trace_v5",
            "evidence.behavior_logprobs",
            "evidence.strict_prefix",
            "evidence.masking",
            "evidence.wire_objects",
            "evidence.artifact_reference",
            "evidence.tito",
        ],
    ),
    (
        "reward",
        &[
            "reward.authority",
            "reward.binding_digest",
            "reward.horizon_quiescence",
            "reward.settlement_window",
            "reward.channels",
        ],
    ),
    ("recovery", &["recovery.restart", "recovery.stale_discard"]),
    (
        "topology",
        &[
            "topology.roster",
            "topology.channels",
            "topology.minimum_roster",
            "topology.opponent_pinning",
        ],
    ),
];

/// Optional clauses may come back unsupported; the run records its fallback.
/// Everything else is mandatory and a rejection stops the run before spend.
pub const CISPO_OPTIONAL_CLAUSES: &[&str] = &[
    "evidence.tito",
    "evidence.artifact_reference",
    "reward.settlement_window",
    "lifecycle.pause_resume",
    "topology.channels",
    "topology.minimum_roster",
    "topology.opponent_pinning",
];

/// Clauses that only apply to a multi-instance topology.
pub const CISPO_TOPOLOGY_ONLY_CLAUSES: &[&str] = &[
    "topology.roster",
    "topology.channels",
    "topology.minimum_roster",
    "topology.opponent_pinning",
];

/// Attempt states, mirrored from `rl_identity.ATTEMPT_STATES`.
pub const CISPO_ATTEMPT_STATES: &[&str] = &[
    "queued",
    "running",
    "awaiting_score",
    "scored",
    "completed",
    "failed",
    "cancelled",
];

/// Mirrored from `rl_identity.TERMINAL_ATTEMPT_STATES`. Exactly one of these
/// may be reported per accepted attempt.
pub const CISPO_TERMINAL_ATTEMPT_STATES: &[&str] = &["completed", "failed", "cancelled"];

/// Mirrored from `rl_records.WIRE_APIS`. The two wires are two datasets, not
/// one: flattening between them is prohibited.
pub const CISPO_WIRE_APIS: &[&str] = &["chat_completions", "responses"];

/// Mirrored from `rl_records.SAMPLING_TRANSPORTS`.
pub const CISPO_SAMPLING_TRANSPORTS: &[&str] = &["message_in_capture_out", "tokens_in_tokens_out"];

/// Mirrored from `rl_identity.HORIZON_KINDS`.
pub const CISPO_HORIZON_KINDS: &[&str] = &["wall_clock", "steps", "env_ticks"];

/// Mirrored from `rl_records.LOGPROB_SENTINEL`. Both a missing-evidence marker
/// and a lower-bound clamp, so receiving it can never prove a real logprob came
/// back.
pub const CISPO_LOGPROB_SENTINEL: f64 = -9999.0;

/// Every clause id, in registry order.
pub fn cispo_all_clauses() -> Vec<&'static str> {
    CISPO_CLAUSE_GROUPS
        .iter()
        .flat_map(|(_, clauses)| clauses.iter().copied())
        .collect()
}

/// Every clause a rejection of which stops the run before session creation.
pub fn cispo_mandatory_clauses() -> Vec<&'static str> {
    cispo_all_clauses()
        .into_iter()
        .filter(|clause| !CISPO_OPTIONAL_CLAUSES.contains(clause))
        .collect()
}

/// The group a clause belongs to, or an error for an unknown clause. Unknown is
/// an error rather than a bucket: a container answering a clause this crate has
/// never heard of is not evidence of agreement.
pub fn cispo_clause_group(clause_id: &str) -> Result<&'static str> {
    for (group, clauses) in CISPO_CLAUSE_GROUPS {
        if clauses.contains(&clause_id) {
            return Ok(group);
        }
    }
    Err(OptimizerError::Container(format!(
        "unknown handshake clause {clause_id:?}"
    )))
}

/// The versioned route table a container advertises under
/// `metadata.optimizer_contracts.cispo`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CispoOptimizerContract {
    pub version: String,
    pub health_route: String,
    pub capabilities_route: String,
    pub handshake_route: String,
    pub taskset_route: String,
    pub taskset_tasks_route: String,
    pub topology_route: Option<String>,
    pub policy_bind_route: String,
    pub policy_set_bind_route: Option<String>,
    pub rollout_route: String,
    pub rollout_state_route: String,
    pub rollout_events_route: String,
    pub rollout_renew_route: String,
    pub rollout_finalize_route: Option<String>,
    pub rollout_terminate_route: String,
    pub trace_route: String,
    pub artifacts_route: Option<String>,
    pub reward_route: String,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl CispoOptimizerContract {
    /// Declared routes the executor requires, paired with their key names.
    pub fn mandatory_routes(&self) -> Vec<(&'static str, &str)> {
        vec![
            ("health_route", self.health_route.as_str()),
            ("capabilities_route", self.capabilities_route.as_str()),
            ("handshake_route", self.handshake_route.as_str()),
            ("taskset_route", self.taskset_route.as_str()),
            ("taskset_tasks_route", self.taskset_tasks_route.as_str()),
            ("policy_bind_route", self.policy_bind_route.as_str()),
            ("rollout_route", self.rollout_route.as_str()),
            ("rollout_state_route", self.rollout_state_route.as_str()),
            ("rollout_events_route", self.rollout_events_route.as_str()),
            ("rollout_renew_route", self.rollout_renew_route.as_str()),
            (
                "rollout_terminate_route",
                self.rollout_terminate_route.as_str(),
            ),
            ("trace_route", self.trace_route.as_str()),
            ("reward_route", self.reward_route.as_str()),
        ]
    }

    /// Declared routes that may be absent. See `CISPO_OPTIONAL_ROUTES` for why
    /// each one is optional.
    pub fn optional_routes(&self) -> Vec<(&'static str, Option<&str>)> {
        vec![
            ("topology_route", self.topology_route.as_deref()),
            (
                "policy_set_bind_route",
                self.policy_set_bind_route.as_deref(),
            ),
            (
                "rollout_finalize_route",
                self.rollout_finalize_route.as_deref(),
            ),
            ("artifacts_route", self.artifacts_route.as_deref()),
        ]
    }

    /// Every mandatory route must be an absolute path. An absent mandatory
    /// route arrives here as the serde default empty string and is rejected by
    /// the same check, so omission and a malformed path fail identically.
    /// Optional routes may be absent, but a declared one must be absolute:
    /// advertising a route the executor cannot address is worse than not
    /// advertising it.
    pub fn validate_routes(&self) -> Result<()> {
        for (name, route) in self.mandatory_routes() {
            if !route.starts_with('/') {
                return Err(OptimizerError::Container(format!(
                    "metadata.optimizer_contracts.cispo.{name} must be an absolute route, got {route:?}"
                )));
            }
        }
        for (name, route) in self.optional_routes() {
            let Some(route) = route else {
                continue;
            };
            if !route.starts_with('/') {
                return Err(OptimizerError::Container(format!(
                    "metadata.optimizer_contracts.cispo.{name} must be an absolute route, got {route:?}"
                )));
            }
        }
        Ok(())
    }

    /// Contract version plus route shape. This is the whole of what reading a
    /// container's advertisement can establish; agreement takes a handshake.
    pub fn validate(&self) -> Result<()> {
        if self.version != CISPO_OPTIMIZER_CONTRACT_VERSION {
            return Err(OptimizerError::Container(format!(
                "container does not advertise metadata.optimizer_contracts.cispo.version={CISPO_OPTIMIZER_CONTRACT_VERSION}"
            )));
        }
        self.validate_routes()
    }

    /// True when the container declares the joint-episode surface: a per-task
    /// topology lookup and an atomic all-instance policy binding.
    pub fn supports_joint_episodes(&self) -> bool {
        self.topology_route.is_some() && self.policy_set_bind_route.is_some()
    }

    /// True when the container declares the staged deferred-scoring path.
    pub fn supports_deferred_finalize(&self) -> bool {
        self.rollout_finalize_route.is_some()
    }
}

/// Canonical content hash over a capability document.
///
/// Excludes `capability_hash` so the document can carry its own hash, sorts
/// keys at every depth, and emits the compact separators the Python side uses,
/// so `sha256:<hex>` here equals `_canonical_sha256` there byte for byte. Any
/// change to any value changes the hash, which is exactly the property that
/// makes a prior preflight — and the handshake built on it — fail closed.
pub fn capability_content_hash(response: &Value) -> Result<String> {
    let Value::Object(object) = response else {
        return Err(OptimizerError::Container(
            "capability response must be an object".to_string(),
        ));
    };
    let mut unhashed = Map::new();
    for (key, value) in object {
        if key == "capability_hash" {
            continue;
        }
        unhashed.insert(key.clone(), value.clone());
    }
    Ok(format!(
        "sha256:{}",
        stable_json_hash(&Value::Object(unhashed))
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn full_contract_value() -> Value {
        json!({
            "version": CISPO_OPTIMIZER_CONTRACT_VERSION,
            "health_route": "/health",
            "capabilities_route": "/training/capabilities",
            "handshake_route": "/training/handshake",
            "taskset_route": "/taskset",
            "taskset_tasks_route": "/taskset/tasks",
            "topology_route": "/topologies/{topology_id}",
            "policy_bind_route": "/policy-configs",
            "policy_set_bind_route": "/policy-sets",
            "rollout_route": "/rollout",
            "rollout_state_route": "/rollouts/{rollout_id}",
            "rollout_events_route": "/rollouts/{rollout_id}/events",
            "rollout_renew_route": "/rollouts/{rollout_id}/renew",
            "rollout_finalize_route": "/rollouts/{rollout_id}/finalize",
            "rollout_terminate_route": "/rollouts/{rollout_id}/terminate",
            "trace_route": "/rollouts/{rollout_id}/trace",
            "artifacts_route": "/rollouts/{rollout_id}/artifacts",
            "reward_route": "/reward"
        })
    }

    fn full_contract() -> CispoOptimizerContract {
        serde_json::from_value(full_contract_value()).expect("contract decodes")
    }

    fn capability_value() -> Value {
        json!({
            "schema_version": CISPO_CAPABILITIES_SCHEMA_VERSION,
            "container_id": "container-0",
            "container_digest": "sha256:aaaa",
            "operations": ["rollout", "reward", "heartbeat"],
            "protocol_versions": ["training.rollout.request.v1"],
            "connection_modes": ["close"],
            "renderer_profile": {
                "profile_id": "renderers.profile.v1",
                "package": "renderers",
                "package_version": "0.1.11",
                "config_digest": "sha256:bbbb",
                "tokenizer_id": "tokenizer-0",
                "tokenizer_digest": "sha256:cccc",
                "stop_token_ids": [200002, 199999],
                "modalities": ["text"],
                "add_generation_prompt": true
            },
            "horizon": {"horizon_kind": "wall_clock", "value_seconds": 5400, "time_dilation": 4.0},
            "lifecycle": {
                "max_concurrency": 30,
                "lease_ttl_seconds": 900.0,
                "straggler_grace_seconds": 120.0,
                "supports_idempotency": true,
                "supports_lease_renewal": true,
                "supports_cancellation": true,
                "supports_exactly_one_terminal": true,
                "supports_event_cursor": true,
                "supports_probe_binding": true
            },
            "evidence": {
                "trace_schema_version": CISPO_TRAINABLE_EPISODE_SCHEMA_VERSION,
                "wire_apis": ["chat_completions"],
                "sampling_transports": ["message_in_capture_out"],
                "token_capture_provenance": ["engine_meta"],
                "supports_trace_v5": true,
                "supports_behavior_logprobs": true,
                "supports_strict_prefix": true,
                "supports_masking": true,
                "supports_wire_objects": true
            },
            "reward": {
                "schema_version": CISPO_REWARD_RECORD_SCHEMA_VERSION,
                "authority": "container",
                "evaluation_plan_id": "plan-0",
                "channels": ["outcome"],
                "supports_binding_digest": true,
                "supports_quiescence_attestation": true,
                "settlement_window_seconds": 150.0
            }
        })
    }

    fn handshake_request(capability_hash: &str) -> CispoHandshakeRequest {
        serde_json::from_value(json!({
            "schema_version": CISPO_HANDSHAKE_SCHEMA_VERSION,
            "run_id": "run-0",
            "optimizer": {"name": "synth_optimizers.cispo", "version": "0.2.20"},
            "policy": {
                "provider": "provider-0",
                "model_id": "model-0",
                "transport": "message_in_capture_out"
            },
            "requirements": ["contract.routes", "evidence.behavior_logprobs"],
            "run_plan": {
                "group_size": 8,
                "groups_per_step": 1,
                "max_execution_slots": 8,
                "maximum_policy_lag": 1,
                "target_train_updates": 10,
                "expected_horizon_seconds": 5400.0
            },
            "taskset": {"taskset_id": "taskset-0", "split": "train", "task_ids": ["task-0"]},
            "clock": {"executor_time": "2026-09-02T00:00:00Z", "monotonic_source": "monotonic"},
            "capability_hash": capability_hash
        }))
        .expect("handshake request decodes")
    }

    /// Every mandatory clause answered `accepted`, so a test can then flip one
    /// clause and know the flip is the only reason the verdict changed.
    fn accepting_verdict(capability_hash: &str, joint_episode: bool) -> CispoHandshakeVerdict {
        let clauses: Vec<Value> = cispo_mandatory_clauses()
            .into_iter()
            .filter(|clause| joint_episode || !CISPO_TOPOLOGY_ONLY_CLAUSES.contains(clause))
            .map(|clause| json!({"clause_id": clause, "verdict": "accepted"}))
            .collect();
        serde_json::from_value(json!({
            "schema_version": CISPO_HANDSHAKE_SCHEMA_VERSION,
            "handshake_id": "hs-0",
            "accepted": true,
            "clauses": clauses,
            "obligations": {
                "max_concurrency": 30,
                "lease_ttl_seconds": 900.0,
                "deferred_scoring": true,
                "quiescence": true,
                "settlement_window_seconds": 150.0,
                "horizon": {"horizon_kind": "wall_clock", "value_seconds": 5400.0}
            },
            "taskset_resolution": [
                {"task_id": "task-0", "content_digest": "sha256:dddd"}
            ],
            "capability_hash": capability_hash,
            "agreement_digest": "sha256:eeee",
            "expires_at": "2026-09-02T01:00:00Z",
            "clock": {"container_time": "2026-09-02T00:00:00Z", "measured_skew_seconds": 0.4}
        }))
        .expect("handshake verdict decodes")
    }

    #[test]
    fn full_contract_validates_every_declared_route() {
        let contract = full_contract();
        contract.validate().expect("full contract validates");
        assert_eq!(
            contract.mandatory_routes().len(),
            CISPO_MANDATORY_ROUTES.len()
        );
        assert_eq!(
            contract.optional_routes().len(),
            CISPO_OPTIONAL_ROUTES.len()
        );
        assert!(contract.supports_joint_episodes());
        assert!(contract.supports_deferred_finalize());
        for (name, _) in contract.mandatory_routes() {
            assert!(
                CISPO_MANDATORY_ROUTES.contains(&name),
                "{name} is not listed in CISPO_MANDATORY_ROUTES"
            );
        }
        for (name, _) in contract.optional_routes() {
            assert!(
                CISPO_OPTIONAL_ROUTES.contains(&name),
                "{name} is not listed in CISPO_OPTIONAL_ROUTES"
            );
        }
    }

    #[test]
    fn each_missing_mandatory_route_is_rejected() {
        for name in CISPO_MANDATORY_ROUTES {
            let mut value = full_contract_value();
            value
                .as_object_mut()
                .expect("object")
                .remove(*name)
                .unwrap_or_else(|| panic!("{name} present in the fixture"));
            let contract: CispoOptimizerContract =
                serde_json::from_value(value).expect("contract decodes without the route");
            let error = contract
                .validate_routes()
                .expect_err(&format!("{name} must be required"));
            let message = error.to_string();
            assert!(
                message.contains(&format!("metadata.optimizer_contracts.cispo.{name}")),
                "error should name the missing route: {message}"
            );
            assert!(
                message.contains("must be an absolute route"),
                "error should keep the shared route error shape: {message}"
            );
        }
    }

    #[test]
    fn each_optional_route_may_be_absent() {
        for name in CISPO_OPTIONAL_ROUTES {
            let mut value = full_contract_value();
            value.as_object_mut().expect("object").remove(*name);
            let contract: CispoOptimizerContract =
                serde_json::from_value(value).expect("contract decodes");
            contract
                .validate()
                .unwrap_or_else(|error| panic!("{name} must be optional: {error}"));
        }
    }

    #[test]
    fn relative_route_is_rejected() {
        for name in CISPO_MANDATORY_ROUTES.iter().chain(CISPO_OPTIONAL_ROUTES) {
            let mut value = full_contract_value();
            value
                .as_object_mut()
                .expect("object")
                .insert((*name).to_string(), json!("rollout"));
            let contract: CispoOptimizerContract =
                serde_json::from_value(value).expect("contract decodes");
            let error = contract
                .validate_routes()
                .expect_err(&format!("{name} must reject a relative path"));
            assert!(
                error.to_string().contains("must be an absolute route"),
                "unexpected error for {name}: {error}"
            );
        }
    }

    #[test]
    fn wrong_contract_version_is_rejected() {
        for version in ["synth_optimizers.cispo.v2", "synth_optimizers.gepa.v2", ""] {
            let mut value = full_contract_value();
            value
                .as_object_mut()
                .expect("object")
                .insert("version".to_string(), json!(version));
            let contract: CispoOptimizerContract =
                serde_json::from_value(value).expect("contract decodes");
            let error = contract
                .validate()
                .expect_err("a foreign contract version must be refused");
            assert!(
                error.to_string().contains(&format!(
                    "metadata.optimizer_contracts.cispo.version={CISPO_OPTIMIZER_CONTRACT_VERSION}"
                )),
                "error should name the required version: {error}"
            );
        }
    }

    #[test]
    fn capability_document_validates_and_hashes() {
        let preflight =
            decode_cispo_capabilities(capability_value()).expect("capability document validates");
        assert!(preflight.capability_hash.starts_with("sha256:"));
        assert_eq!(preflight.capabilities.advertised_concurrency(), 30);
        // The note spells the horizon magnitude `value_seconds`; the record
        // calls it `value`. The alias must land on the record's field.
        assert_eq!(
            preflight
                .capabilities
                .horizon
                .as_ref()
                .expect("horizon")
                .value,
            5400.0
        );
    }

    #[test]
    fn capability_hash_is_stable_across_key_reordering() {
        let ordered = capability_value();
        let reordered = {
            let object = ordered.as_object().expect("object").clone();
            let mut keys: Vec<String> = object.keys().cloned().collect();
            keys.reverse();
            let mut shuffled = Map::new();
            for key in keys {
                let value = object.get(&key).expect("key").clone();
                shuffled.insert(key, value);
            }
            Value::Object(shuffled)
        };
        let first = capability_content_hash(&ordered).expect("hash");
        let second = capability_content_hash(&reordered).expect("hash");
        assert_eq!(
            first, second,
            "the hash must not depend on key order at any depth"
        );
        // An advertised hash is verified against the computed one, so a
        // document that carries its own hash still hashes to the same value.
        let mut self_hashed = ordered.clone();
        self_hashed
            .as_object_mut()
            .expect("object")
            .insert("capability_hash".to_string(), json!(first.clone()));
        assert_eq!(
            capability_content_hash(&self_hashed).expect("hash"),
            first,
            "capability_hash must be excluded from its own hash"
        );
        decode_cispo_capabilities(self_hashed).expect("a self-consistent hash is accepted");
    }

    #[test]
    fn capability_hash_changes_on_any_value_change() {
        let baseline = capability_content_hash(&capability_value()).expect("hash");
        // Each entry is a path into the document and the value to write there.
        // The last one adds a key the crate does not know: an addition must
        // invalidate the preflight just as a change does.
        let mutations: &[(&[&str], Value)] = &[
            (&["container_digest"], json!("sha256:ffff")),
            (&["lifecycle", "max_concurrency"], json!(29)),
            (&["renderer_profile", "stop_token_ids"], json!([200002])),
            (&["evidence", "supports_behavior_logprobs"], json!(false)),
            (&["reward", "settlement_window_seconds"], json!(151.0)),
            (&["supports_live_frames"], json!(true)),
        ];
        for (path, replacement) in mutations {
            let mut value = capability_value();
            let mut cursor = &mut value;
            for key in &path[..path.len() - 1] {
                cursor = cursor
                    .get_mut(*key)
                    .unwrap_or_else(|| panic!("{key} present in the fixture"));
            }
            cursor
                .as_object_mut()
                .expect("object")
                .insert(path[path.len() - 1].to_string(), replacement.clone());
            let mutated = capability_content_hash(&value).expect("hash");
            assert_ne!(
                baseline,
                mutated,
                "changing {} must invalidate the prior preflight",
                path.join(".")
            );
        }
    }

    #[test]
    fn capability_document_with_a_disagreeing_advertised_hash_is_refused() {
        let mut value = capability_value();
        value
            .as_object_mut()
            .expect("object")
            .insert("capability_hash".to_string(), json!("sha256:not-the-hash"));
        let error = decode_cispo_capabilities(value).expect_err("a stale hash must be refused");
        assert!(
            error.to_string().contains("capability_hash mismatch"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn handshake_accepts_when_every_mandatory_clause_is_accepted() {
        let preflight = decode_cispo_capabilities(capability_value()).expect("capabilities");
        let request = handshake_request(&preflight.capability_hash);
        request.validate().expect("request validates");
        let verdict = accepting_verdict(&preflight.capability_hash, false);
        verdict
            .validate(&request, false)
            .expect("an all-accepted verdict is agreement");
        assert!(verdict.blocking_clauses(false).is_empty());
        assert!(verdict.fallback_clauses().is_empty());
        assert!(verdict.degraded_clauses().is_empty());
        verdict
            .assert_clock_skew_within(1.0)
            .expect("skew is inside tolerance");
        verdict
            .assert_clock_skew_within(0.1)
            .expect_err("skew beyond tolerance is a rejection");
    }

    #[test]
    fn rejected_mandatory_clause_is_distinguished_from_unsupported_optional_clause() {
        let preflight = decode_cispo_capabilities(capability_value()).expect("capabilities");
        let request = handshake_request(&preflight.capability_hash);

        // An optional clause coming back unsupported records a fallback and the
        // run proceeds.
        let mut tolerated = accepting_verdict(&preflight.capability_hash, false);
        tolerated.clauses.push(HandshakeClauseVerdict {
            clause_id: "evidence.tito".to_string(),
            verdict: ClauseVerdict::Unsupported,
            ..HandshakeClauseVerdict::default()
        });
        tolerated
            .validate(&request, false)
            .expect("an unsupported optional clause is a fallback, not a stop");
        assert!(tolerated.blocking_clauses(false).is_empty());
        assert_eq!(tolerated.fallback_clauses(), vec!["evidence.tito"]);

        // The same verdict value on a mandatory clause stops the run before
        // session creation.
        let mut blocked = accepting_verdict(&preflight.capability_hash, false);
        for clause in &mut blocked.clauses {
            if clause.clause_id == "reward.horizon_quiescence" {
                clause.verdict = ClauseVerdict::Rejected;
                clause.reason = Some("cannot stop policy-authored background work".to_string());
            }
        }
        assert_eq!(
            blocked.blocking_clauses(false),
            vec!["reward.horizon_quiescence"]
        );
        let error = blocked
            .validate(&request, false)
            .expect_err("a rejected mandatory clause must stop the run");
        assert!(
            error.to_string().contains("reward.horizon_quiescence"),
            "the error must name the clause list: {error}"
        );
        assert!(blocked.fallback_clauses().is_empty());

        // A verdict that simply omits a mandatory clause is blocked too:
        // silence is not agreement.
        let mut silent = accepting_verdict(&preflight.capability_hash, false);
        silent
            .clauses
            .retain(|clause| clause.clause_id != "evidence.trace_v5");
        assert_eq!(silent.blocking_clauses(false), vec!["evidence.trace_v5"]);

        // A degraded clause is neither: it is an instruction to lower the plan
        // and ask again.
        let mut degraded = accepting_verdict(&preflight.capability_hash, false);
        for clause in &mut degraded.clauses {
            if clause.clause_id == "lifecycle.concurrency" {
                clause.verdict = ClauseVerdict::Degraded;
            }
        }
        assert_eq!(degraded.degraded_clauses(), vec!["lifecycle.concurrency"]);
        assert!(degraded.blocking_clauses(false).is_empty());
    }

    #[test]
    fn topology_only_clauses_are_required_only_for_a_joint_episode() {
        let preflight = decode_cispo_capabilities(capability_value()).expect("capabilities");
        let request = handshake_request(&preflight.capability_hash);
        let single = accepting_verdict(&preflight.capability_hash, false);
        single
            .validate(&request, false)
            .expect("a single-instance run does not need the topology clauses");
        let mut blocking = single.blocking_clauses(true);
        blocking.sort_unstable();
        let mut expected: Vec<String> = CISPO_TOPOLOGY_ONLY_CLAUSES
            .iter()
            .filter(|clause| !CISPO_OPTIONAL_CLAUSES.contains(clause))
            .map(|clause| (*clause).to_string())
            .collect();
        expected.sort_unstable();
        assert_eq!(blocking, expected);
        accepting_verdict(&preflight.capability_hash, true)
            .validate(&request, true)
            .expect("a joint episode answering the topology clauses is agreement");
    }

    #[test]
    fn handshake_verdict_on_a_changed_capability_document_is_refused() {
        let preflight = decode_cispo_capabilities(capability_value()).expect("capabilities");
        let request = handshake_request(&preflight.capability_hash);
        let verdict = accepting_verdict("sha256:some-other-document", false);
        let error = verdict
            .validate(&request, false)
            .expect_err("a verdict built on another capability document must be refused");
        assert!(
            error.to_string().contains("capability_hash"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn clause_registry_matches_the_shared_definition() {
        let all = cispo_all_clauses();
        // The shared clause registry carries 34 clauses across 8 groups. A
        // change here is a change to the Python half too, never only to this.
        assert_eq!(all.len(), 34, "clause registry changed: {all:?}");
        assert_eq!(CISPO_CLAUSE_GROUPS.len(), 8);
        assert_eq!(
            cispo_mandatory_clauses().len(),
            all.len() - CISPO_OPTIONAL_CLAUSES.len()
        );
        for clause in CISPO_OPTIONAL_CLAUSES {
            assert!(all.contains(clause), "{clause} is not a known clause");
            assert!(!cispo_mandatory_clauses().contains(clause));
        }
        for clause in &all {
            cispo_clause_group(clause).expect("every clause has a group");
        }
        cispo_clause_group("contract.unknown").expect_err("an unknown clause is an error");
    }

    #[test]
    fn rollout_submission_and_state_round_trip() {
        let submission: CispoRolloutSubmission = serde_json::from_value(json!({
            "idempotency_key": "run-0:group-0:0",
            "handshake_id": "hs-0",
            "agreement_digest": "sha256:eeee",
            "run_id": "run-0",
            "group_id": "group-0",
            "sample_index": 0,
            "seed": 7,
            "policy_revision": 17,
            "behavior_fingerprint": "sha256:ffff",
            "task_id": "task-0",
            "policy_set_revision": "set-20"
        }))
        .expect("submission decodes");
        submission.validate().expect("submission validates");
        // The note's `policy_set_revision` must land on the record's
        // `policy_set_revision_id`.
        assert_eq!(submission.policy_set_revision_id.as_deref(), Some("set-20"));

        let ack = decode_cispo_rollout_ack(json!({
            "rollout_id": "rollout-0",
            "idempotency_key": "run-0:group-0:0",
            "state": "queued",
            "lease_expires_at": "2026-09-02T00:15:00Z"
        }))
        .expect("ack decodes");
        ack.validate_for(&submission).expect("ack echoes the key");

        let state = decode_cispo_rollout_state(json!({
            "rollout_id": "rollout-0",
            "state": "running",
            "lease_expires_at": "2026-09-02T00:15:00Z",
            "event_cursor": "8"
        }))
        .expect("state decodes");
        assert!(!state.is_terminal());

        decode_cispo_rollout_state(json!({
            "rollout_id": "rollout-0",
            "state": "running"
        }))
        .expect_err("active work with no lease is not recoverable");
        decode_cispo_rollout_state(json!({
            "rollout_id": "rollout-0",
            "state": "sprinting"
        }))
        .expect_err("an unknown attempt state is refused");
    }

    #[test]
    fn trace_reference_requires_a_digest_and_a_way_to_read_it() {
        decode_cispo_trace_reference(json!({
            "rollout_id": "rollout-0",
            "trace_digest": "sha256:dddd",
            "schema_version": CISPO_TRAINABLE_EPISODE_SCHEMA_VERSION,
            "uri": "https://example.invalid/trace",
            "size_bytes": 2_000_000_000u64
        }))
        .expect("a reference plus digest is valid evidence");
        decode_cispo_trace_reference(json!({
            "rollout_id": "rollout-0",
            "schema_version": CISPO_TRAINABLE_EPISODE_SCHEMA_VERSION,
            "inline": {"segments": []}
        }))
        .expect_err("an unsealed trace is an evidence failure");
        decode_cispo_trace_reference(json!({
            "rollout_id": "rollout-0",
            "trace_digest": "sha256:dddd",
            "schema_version": CISPO_TRAINABLE_EPISODE_SCHEMA_VERSION
        }))
        .expect_err("a trace that is neither inline nor resolvable is unusable");
    }

    #[test]
    fn reward_receipt_is_bound_to_the_rollout_and_the_trace_digest() {
        let receipt_value = json!({
            "reward_id": "reward-0",
            "rollout_id": "rollout-0",
            "trace_digest": "sha256:dddd",
            "channels": [
                {"channel_id": "outcome", "team_id": null, "measure": 0.0, "rank": 1}
            ],
            "optimized_channel": "outcome",
            "terminal_status": "completed",
            "evaluation_plan_id": "plan-0",
            "horizon": {
                "horizon_kind": "wall_clock",
                "horizon_value": 5400.0,
                "scored_at_offset_seconds": 12.0,
                "clipped": false,
                "quiescence_attested": true,
                "settlement_window_seconds": 150.0
            },
            "schema_version": CISPO_REWARD_RECORD_SCHEMA_VERSION
        });
        let receipt = decode_cispo_reward_receipt(receipt_value.clone(), Some("sha256:dddd"))
            .expect("receipt validates against its episode");
        // Zero stays distinguishable from absent.
        assert_eq!(receipt.value(None).expect("optimized channel"), 0.0);
        decode_cispo_reward_receipt(receipt_value.clone(), Some("sha256:other"))
            .expect_err("a receipt bound to another trace is refused");

        let mut channelless = receipt_value.clone();
        channelless.as_object_mut().expect("object")["channels"] = json!([]);
        decode_cispo_reward_receipt(channelless, None).expect_err("absent is not zero");

        let mut unquiesced = receipt_value;
        unquiesced.as_object_mut().expect("object")["horizon"]["quiescence_attested"] =
            json!(false);
        decode_cispo_reward_receipt(unquiesced, None)
            .expect_err("no attestation and no clipping is a reward-integrity failure");
    }
}
