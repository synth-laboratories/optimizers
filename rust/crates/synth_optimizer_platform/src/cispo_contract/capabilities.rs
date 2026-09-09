//! The hashed capability document and the declared facts inside it.
//!
//! Reading this document is discovery, not agreement: it says what a container
//! can do, never that it can honor a particular run. Persist the whole response
//! and its hash in every run receipt, because the hash is what makes the
//! execution contract auditable later and what makes a changed contract fail
//! closed.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{
    capability_content_hash, CISPO_CAPABILITIES_SCHEMA_VERSION, CISPO_HORIZON_KINDS,
    CISPO_RENDERER_PROFILE_SCHEMA_VERSION, CISPO_SAMPLING_TRANSPORTS, CISPO_WIRE_APIS,
};
use crate::cache::stable_json_hash;
use crate::container_contract::JsonMap;
use crate::error::{OptimizerError, Result};

/// Pinned renderer identity. A version string alone is not an identity, which
/// is why every field below is part of the fingerprint.
///
/// Mirrors `rl_records.RendererProfile`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RendererProfileDoc {
    pub profile_id: String,
    pub package: String,
    pub package_version: String,
    pub config_digest: String,
    pub tokenizer_id: String,
    pub tokenizer_digest: String,
    pub stop_token_ids: Vec<i64>,
    pub modalities: Vec<String>,
    pub add_generation_prompt: Option<bool>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl RendererProfileDoc {
    pub fn validate(&self) -> Result<()> {
        if self.profile_id.trim().is_empty() {
            return Err(OptimizerError::Container(
                "renderer profile_id is required".to_string(),
            ));
        }
        if self.stop_token_ids.is_empty() {
            return Err(OptimizerError::Container(
                "renderer profile must declare stop token ids".to_string(),
            ));
        }
        Ok(())
    }

    /// Digest of everything that changes what a token sequence means.
    ///
    /// Equality is one comparison rather than a field walk at the call site, so
    /// a newly added field cannot be forgotten by one of several comparisons.
    pub fn fingerprint(&self) -> String {
        let modalities = if self.modalities.is_empty() {
            vec!["text".to_string()]
        } else {
            self.modalities.clone()
        };
        stable_json_hash(&serde_json::json!({
            "schema_version": CISPO_RENDERER_PROFILE_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "package": self.package,
            "package_version": self.package_version,
            "config_digest": self.config_digest,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_digest": self.tokenizer_digest,
            "stop_token_ids": self.stop_token_ids,
            "modalities": modalities,
            "add_generation_prompt": self.add_generation_prompt.unwrap_or(true),
        }))
    }

    /// The binding's profile must equal the training session's profile. A
    /// mismatch is a preflight failure before any paid request, and the same
    /// mismatch found at evaluation time is an evidence failure.
    pub fn assert_matches(&self, other: &RendererProfileDoc) -> Result<()> {
        let mine = self.fingerprint();
        let theirs = other.fingerprint();
        if mine != theirs {
            return Err(OptimizerError::Container(format!(
                "renderer profile mismatch: {}@{mine} != {}@{theirs}",
                self.profile_id, other.profile_id
            )));
        }
        Ok(())
    }
}

/// Mirrors `rl_identity.Horizon`.
///
/// The Python record calls the magnitude `value`; the design note's JSON writes
/// `value_seconds`. The Python name is authoritative because the field also
/// carries step and tick horizons, where "seconds" would be a lie. The note's
/// spelling is accepted as an alias so an already-deployed container keeps
/// decoding.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HorizonDoc {
    pub horizon_kind: String,
    #[serde(alias = "value_seconds")]
    pub value: f64,
    pub time_dilation: Option<f64>,
    pub grace_seconds: Option<f64>,
    /// A step or tick horizon carries no duration of its own, so leases and
    /// queue timeouts cannot be derived from it without a declared conversion.
    pub seconds_per_unit: Option<f64>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl HorizonDoc {
    pub fn validate(&self) -> Result<()> {
        if !CISPO_HORIZON_KINDS.contains(&self.horizon_kind.as_str()) {
            return Err(OptimizerError::Container(format!(
                "unknown horizon_kind {:?}",
                self.horizon_kind
            )));
        }
        if !(self.value.is_finite() && self.value > 0.0) {
            return Err(OptimizerError::Container(
                "horizon value must be positive".to_string(),
            ));
        }
        let seconds_per_unit = self.seconds_per_unit.unwrap_or(1.0);
        if !(seconds_per_unit.is_finite() && seconds_per_unit > 0.0) {
            return Err(OptimizerError::Container(
                "seconds_per_unit must be positive".to_string(),
            ));
        }
        Ok(())
    }

    /// Wall-clock budget a lease must cover, including the declared grace. An
    /// hour-scale episode is a normal case; a queue that assumes minute-scale
    /// attempts will declare healthy work dead, so this is derived and never
    /// guessed.
    pub fn lease_seconds(&self) -> f64 {
        let grace = self.grace_seconds.unwrap_or(0.0);
        if self.horizon_kind == "wall_clock" {
            return self.value + grace;
        }
        self.value * self.seconds_per_unit.unwrap_or(1.0) + grace
    }
}

/// Mirrors `rl_identity.AgentInstance`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentInstanceDoc {
    pub agent_instance_id: String,
    pub role_id: String,
    pub policy_type_id: String,
    pub team_id: String,
    pub trainable: bool,
    pub pinned_identity: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// Mirrors `rl_identity.Team`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TeamDoc {
    pub team_id: String,
    pub trainable: bool,
    pub minimum_viable_roster: Option<u32>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// Mirrors `rl_identity.CommunicationChannel`. Another instance's message
/// tokens are observation, never free reward, so `trainable_for_author` names
/// the author's side only.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CommunicationChannelDoc {
    pub channel_id: String,
    pub scope: String,
    pub trainable_for_author: Option<bool>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// Mirrors `rl_identity.Topology`. A container-declared roster: the executor
/// binds it and never infers it from an agent count, a role string, or a task
/// name.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TopologyDoc {
    pub topology_id: String,
    pub turn_model: String,
    pub actuation_model: String,
    pub reward_relation: String,
    pub agent_instances: Vec<AgentInstanceDoc>,
    pub teams: Vec<TeamDoc>,
    pub communication_channels: Vec<CommunicationChannelDoc>,
    pub horizon: Option<HorizonDoc>,
    pub parameter_groups: JsonMap,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl TopologyDoc {
    pub fn validate(&self) -> Result<()> {
        if self.agent_instances.is_empty() {
            return Err(OptimizerError::Container(
                "topology declares no agent instances".to_string(),
            ));
        }
        let mut seen: Vec<&str> = Vec::new();
        for instance in &self.agent_instances {
            if instance.agent_instance_id.trim().is_empty() {
                return Err(OptimizerError::Container(
                    "agent_instance_id is required".to_string(),
                ));
            }
            if seen.contains(&instance.agent_instance_id.as_str()) {
                return Err(OptimizerError::Container(
                    "duplicate agent_instance_id in topology".to_string(),
                ));
            }
            seen.push(instance.agent_instance_id.as_str());
            // A non-trainable instance is an opponent. Reproducibility needs
            // its identity even though no trainable evidence comes back for it.
            if !instance.trainable
                && instance
                    .pinned_identity
                    .as_deref()
                    .unwrap_or_default()
                    .trim()
                    .is_empty()
            {
                return Err(OptimizerError::Container(format!(
                    "non-trainable instance {} must pin an immutable identity",
                    instance.agent_instance_id
                )));
            }
            if !self
                .teams
                .iter()
                .any(|team| team.team_id == instance.team_id)
            {
                return Err(OptimizerError::Container(format!(
                    "instance {} names undeclared team {:?}",
                    instance.agent_instance_id, instance.team_id
                )));
            }
        }
        // A concurrent real-time topology is a first-class case, and its leases
        // and queue timeouts are derived from the horizon rather than guessed.
        if self.turn_model == "concurrent_realtime" && self.horizon.is_none() {
            return Err(OptimizerError::Container(
                "a concurrent real-time topology must declare a horizon".to_string(),
            ));
        }
        if let Some(horizon) = &self.horizon {
            horizon.validate()?;
        }
        Ok(())
    }

    /// True when the topology needs the joint-episode surface: an atomic
    /// all-instance binding and the topology-scoped clauses.
    pub fn is_joint(&self) -> bool {
        self.agent_instances.len() > 1
    }

    pub fn trainable_instances(&self) -> Vec<&AgentInstanceDoc> {
        self.agent_instances
            .iter()
            .filter(|i| i.trainable)
            .collect()
    }

    pub fn opponent_instances(&self) -> Vec<&AgentInstanceDoc> {
        self.agent_instances
            .iter()
            .filter(|i| !i.trainable)
            .collect()
    }
}

/// Lifecycle capability flags and the numbers derived from them. Leases,
/// heartbeats, and queue timeouts come from `lease_ttl_seconds` and the
/// advertised horizon; none of them is a constant in the engine.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LifecycleCapabilities {
    pub max_concurrency: u32,
    pub lease_ttl_seconds: f64,
    pub straggler_grace_seconds: f64,
    pub supports_idempotency: bool,
    pub supports_lease_renewal: bool,
    pub supports_cancellation: bool,
    pub supports_exactly_one_terminal: bool,
    pub supports_event_cursor: bool,
    pub supports_event_stream: bool,
    pub supports_deferred_scoring: bool,
    pub supports_pause_resume: bool,
    pub supports_checkpoint_resume: bool,
    /// A binding kind returning deterministic, explicitly synthetic evidence,
    /// so the whole evidence path can be walked at zero provider cost. A
    /// container without it costs one real canary attempt instead.
    pub supports_probe_binding: bool,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// Evidence capability flags. These are what makes a trace trainable, so every
/// one of them is a claim the probe episode has to make good on.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct EvidenceCapabilities {
    pub trace_schema_version: String,
    pub wire_apis: Vec<String>,
    pub sampling_transports: Vec<String>,
    pub token_capture_provenance: Vec<String>,
    pub supports_trace_v5: bool,
    pub supports_behavior_logprobs: bool,
    pub supports_strict_prefix: bool,
    pub supports_masking: bool,
    pub supports_wire_objects: bool,
    pub supports_artifact_reference: bool,
    /// `tokens_in_tokens_out`. Optional, and never the route by which a second
    /// renderer enters the run: a container declaring it must declare the
    /// identical renderer profile.
    pub supports_tokens_in_tokens_out: bool,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl EvidenceCapabilities {
    /// Reject a declared wire or transport this crate does not know, rather
    /// than reading the unknown value as absent.
    pub fn validate(&self) -> Result<()> {
        for wire in &self.wire_apis {
            if !CISPO_WIRE_APIS.contains(&wire.as_str()) {
                return Err(OptimizerError::Container(format!(
                    "unknown wire_api {wire:?}"
                )));
            }
        }
        for transport in &self.sampling_transports {
            if !CISPO_SAMPLING_TRANSPORTS.contains(&transport.as_str()) {
                return Err(OptimizerError::Container(format!(
                    "unknown sampling_transport {transport:?}"
                )));
            }
        }
        if self.supports_tokens_in_tokens_out
            && !self
                .sampling_transports
                .iter()
                .any(|t| t == "tokens_in_tokens_out")
        {
            return Err(OptimizerError::Container(
                "container declares tokens_in_tokens_out but does not list it as a sampling transport"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Reward authority. The reward is the container's to state; the executor reads
/// a receipt and never recomputes one.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RewardAuthorityCapabilities {
    pub schema_version: String,
    /// Who computes the reward. Anything but the container is a refusal: an
    /// executor-computed reward is not a container-authoritative one.
    pub authority: String,
    pub evaluation_plan_id: String,
    pub channels: Vec<String>,
    pub reward_relation: Option<String>,
    pub supports_binding_digest: bool,
    pub supports_quiescence_attestation: bool,
    pub supports_horizon_clipping: bool,
    pub supports_deferred_scoring: bool,
    pub settlement_window_seconds: f64,
    pub horizon: Option<HorizonDoc>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl RewardAuthorityCapabilities {
    pub fn validate(&self) -> Result<()> {
        if self.authority != "container" {
            return Err(OptimizerError::Container(format!(
                "reward authority must be the container, got {:?}",
                self.authority
            )));
        }
        if self.evaluation_plan_id.trim().is_empty() {
            return Err(OptimizerError::Container(
                "reward authority must declare a stable evaluation_plan_id".to_string(),
            ));
        }
        if self.channels.is_empty() {
            return Err(OptimizerError::Container(
                "reward authority must declare at least one channel".to_string(),
            ));
        }
        // Neither a quiescence attestation nor a horizon-clipped snapshot means
        // post-horizon activity can reach the reward, which silently rewrites
        // the ranking.
        if !self.supports_quiescence_attestation && !self.supports_horizon_clipping {
            return Err(OptimizerError::Container(
                "reward authority declares neither a quiescence attestation nor horizon clipping"
                    .to_string(),
            ));
        }
        if let Some(horizon) = &self.horizon {
            horizon.validate()?;
        }
        Ok(())
    }
}

/// The hashed capability document.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CispoCapabilityResponse {
    pub schema_version: String,
    /// The container's own hash of this document, excluding this field. Absent
    /// is allowed on the wire; a present value disagreeing with the computed
    /// hash is a refusal.
    pub capability_hash: Option<String>,
    pub container_id: String,
    pub container_digest: String,
    pub container_version: Option<String>,
    pub contract_version: Option<String>,
    pub operations: Vec<String>,
    pub protocol_versions: Vec<String>,
    pub connection_modes: Vec<String>,
    pub renderer_profile: Option<RendererProfileDoc>,
    pub topology: Option<TopologyDoc>,
    pub horizon: Option<HorizonDoc>,
    pub lifecycle: LifecycleCapabilities,
    pub evidence: EvidenceCapabilities,
    pub reward: RewardAuthorityCapabilities,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl CispoCapabilityResponse {
    /// Content hash of this document as a typed value.
    ///
    /// Prefer `capability_content_hash` over the raw response when one is at
    /// hand: a typed round-trip can renumber an integer as a float, and the
    /// hash the container computed is over the bytes it sent.
    pub fn content_hash(&self) -> Result<String> {
        capability_content_hash(&serde_json::to_value(self)?)
    }

    /// The advertised hash must equal the computed one. Returns the hash the
    /// run should record.
    pub fn verify_advertised_hash(&self, computed: &str) -> Result<String> {
        match self.capability_hash.as_deref() {
            Some(advertised) if advertised != computed => Err(OptimizerError::Container(format!(
                "capability_hash mismatch: container advertised {advertised:?}, document hashes to {computed:?}"
            ))),
            _ => Ok(computed.to_string()),
        }
    }

    /// Everything reading the capability document can establish on its own.
    /// Requirement satisfaction — group size against concurrency, horizon
    /// against lease TTL, renderer profile against the training session — is
    /// the handshake's job, not this function's.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != CISPO_CAPABILITIES_SCHEMA_VERSION {
            return Err(OptimizerError::Container(format!(
                "container does not advertise capability schema_version={CISPO_CAPABILITIES_SCHEMA_VERSION}"
            )));
        }
        for (name, value) in [
            ("container_id", self.container_id.as_str()),
            ("container_digest", self.container_digest.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(OptimizerError::Container(format!(
                    "capability response must include {name}"
                )));
            }
        }
        if self.lifecycle.max_concurrency == 0 {
            return Err(OptimizerError::Container(
                "capability response must advertise a positive lifecycle.max_concurrency"
                    .to_string(),
            ));
        }
        if let Some(profile) = &self.renderer_profile {
            profile.validate()?;
        }
        if let Some(topology) = &self.topology {
            topology.validate()?;
        }
        if let Some(horizon) = &self.horizon {
            horizon.validate()?;
        }
        self.evidence.validate()?;
        self.reward.validate()?;
        Ok(())
    }

    /// Concurrency the executor may actually use, which is the advertised
    /// maximum and never the requested one.
    pub fn advertised_concurrency(&self) -> u32 {
        self.lifecycle.max_concurrency
    }

    /// Whether the declared roster needs the joint-episode surface.
    pub fn is_joint_episode(&self) -> bool {
        self.topology.as_ref().is_some_and(TopologyDoc::is_joint)
    }
}

/// What a preflight keeps: the typed document, the hash the run records, and
/// the received document the hash was taken over.
#[derive(Clone, Debug)]
pub struct CispoCapabilityPreflight {
    pub capabilities: CispoCapabilityResponse,
    pub capability_hash: String,
    pub document: Value,
}

/// Decode, hash, and validate a capability response. Fail-closed: the hash is
/// computed from the received document rather than from a typed round-trip, and
/// a container-advertised hash that disagrees is a refusal.
pub fn decode_cispo_capabilities(value: Value) -> Result<CispoCapabilityPreflight> {
    let computed = capability_content_hash(&value)?;
    let capabilities: CispoCapabilityResponse = serde_json::from_value(value.clone())?;
    capabilities.validate()?;
    let capability_hash = capabilities.verify_advertised_hash(&computed)?;
    Ok(CispoCapabilityPreflight {
        capabilities,
        capability_hash,
        document: value,
    })
}
