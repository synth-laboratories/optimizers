//! The two-sided readiness agreement.
//!
//! A container that publishes a compliant contract can still be unable to
//! honor this particular run: its concurrency may be under the requested group
//! size, its lease TTL shorter than the horizon, its renderer profile a
//! different build, its taskset rows changed since the config was written, its
//! clock skewed against the horizon the reward will be read at. Each of those
//! produces a run that starts successfully and wastes spend before failing, or
//! worse, trains on evidence that was never valid. So the container answers per
//! clause, and nothing is negotiated after training starts.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::capabilities::{HorizonDoc, RendererProfileDoc};
use super::{
    cispo_clause_group, cispo_mandatory_clauses, CISPO_HANDSHAKE_SCHEMA_VERSION,
    CISPO_OPTIONAL_CLAUSES, CISPO_SAMPLING_TRANSPORTS, CISPO_TOPOLOGY_ONLY_CLAUSES,
};
use crate::container_contract::JsonMap;
use crate::error::{OptimizerError, Result};

/// Identity of the asking optimizer. Present so the container can record who it
/// agreed with, never so it can behave differently.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HandshakeOptimizerIdentity {
    pub name: String,
    pub version: String,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// The policy the run will sample from. `transport` is one of
/// `CISPO_SAMPLING_TRANSPORTS`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HandshakePolicyRequest {
    pub provider: String,
    pub model_id: String,
    pub transport: String,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// Topology the executor expects to bind. It accepts a declared topology by id
/// and never defines one.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HandshakeTopologyRequest {
    pub expected_topology_id: Option<String>,
    pub trainable_teams: Vec<String>,
    pub partial_roster: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// The run plan the container is being asked to honor. A degraded clause is
/// accepted only by lowering these numbers and re-handshaking, never by
/// assuming the lowered plan.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HandshakeRunPlanRequest {
    pub group_size: u32,
    pub groups_per_step: u32,
    pub max_execution_slots: u32,
    pub maximum_policy_lag: u32,
    pub target_train_updates: u32,
    pub expected_horizon_seconds: f64,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// Task selection, resolved by discovery rather than configured. Task ids are
/// discovered and persisted; there is no allowlist.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HandshakeTasksetRequest {
    pub taskset_id: String,
    pub split: String,
    pub task_ids: Vec<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// Executor-side clock. For a wall-clock horizon both sides record their time,
/// because the horizon is the instant the reward is read.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HandshakeExecutorClock {
    pub executor_time: String,
    pub monotonic_source: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// The executor's requirement document.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CispoHandshakeRequest {
    pub schema_version: String,
    pub run_id: String,
    pub optimizer: HandshakeOptimizerIdentity,
    pub policy: HandshakePolicyRequest,
    pub renderer_profile: Option<RendererProfileDoc>,
    /// Clause ids the run requires. Every entry must be a known clause.
    pub requirements: Vec<String>,
    pub topology: Option<HandshakeTopologyRequest>,
    pub run_plan: HandshakeRunPlanRequest,
    pub taskset: HandshakeTasksetRequest,
    pub clock: HandshakeExecutorClock,
    /// The capability hash this ask was built on. The verdict must echo it, so
    /// a capability document that changed under the preflight is caught.
    pub capability_hash: String,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl CispoHandshakeRequest {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != CISPO_HANDSHAKE_SCHEMA_VERSION {
            return Err(OptimizerError::Container(format!(
                "handshake request schema_version must be {CISPO_HANDSHAKE_SCHEMA_VERSION:?}, got {:?}",
                self.schema_version
            )));
        }
        if self.run_id.trim().is_empty() {
            return Err(OptimizerError::Container(
                "handshake request must include run_id".to_string(),
            ));
        }
        if self.capability_hash.trim().is_empty() {
            return Err(OptimizerError::Container(
                "handshake request must name the capability_hash it was built on".to_string(),
            ));
        }
        if !CISPO_SAMPLING_TRANSPORTS.contains(&self.policy.transport.as_str()) {
            return Err(OptimizerError::Container(format!(
                "unknown sampling_transport {:?}",
                self.policy.transport
            )));
        }
        if self.run_plan.group_size == 0 {
            return Err(OptimizerError::Container(
                "handshake request must ask for a positive group_size".to_string(),
            ));
        }
        for clause in &self.requirements {
            cispo_clause_group(clause)?;
        }
        Ok(())
    }
}

/// The four verdict values. A bare boolean tells you a run will fail without
/// telling you what to change, so there is no boolean here.
///
/// `Default` is `Rejected` on purpose: a verdict that failed to arrive must not
/// read as agreement. Deserialization of an unrecognized verdict fails rather
/// than falling back, for the same reason.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClauseVerdict {
    Accepted,
    Degraded,
    /// The default. Silence is not agreement.
    #[default]
    Rejected,
    Unsupported,
}

impl ClauseVerdict {
    /// Stops the run before session creation when the clause is mandatory. An
    /// unsupported mandatory clause is as blocking as a rejected one: the
    /// difference is why, not whether.
    pub fn blocks_mandatory(self) -> bool {
        matches!(self, Self::Rejected | Self::Unsupported)
    }

    /// Records a fallback when the clause is optional.
    pub fn needs_fallback(self) -> bool {
        matches!(self, Self::Rejected | Self::Unsupported)
    }

    /// Acceptable only by lowering the run plan and re-handshaking.
    pub fn needs_replan(self) -> bool {
        matches!(self, Self::Degraded)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Degraded => "degraded",
            Self::Rejected => "rejected",
            Self::Unsupported => "unsupported",
        }
    }
}

/// One clause's answer. `reason` exists so a degraded or rejected clause names
/// what to change.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HandshakeClauseVerdict {
    pub clause_id: String,
    pub verdict: ClauseVerdict,
    pub reason: Option<String>,
    pub note: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// What the container commits to. These are the numbers the queue engine
/// derives leases, timeouts, and admission from.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HandshakeObligations {
    pub max_concurrency: u32,
    pub lease_ttl_seconds: f64,
    pub deferred_scoring: bool,
    pub quiescence: bool,
    pub settlement_window_seconds: f64,
    pub horizon: Option<HorizonDoc>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// One resolved task row. Field names follow `rl_identity.TaskSpec`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TasksetResolutionEntry {
    pub task_id: String,
    pub content_digest: String,
    pub topology_ref: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// Container-side clock and the skew both sides measured.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HandshakeContainerClock {
    pub container_time: String,
    pub measured_skew_seconds: f64,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// The container's per-clause answer. Every rollout carries `handshake_id`, and
/// the container must refuse any attempt whose handshake is absent, expired,
/// revoked, or whose `agreement_digest` does not match, so a run cannot drift
/// out from under its own agreement.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CispoHandshakeVerdict {
    pub schema_version: String,
    pub handshake_id: String,
    pub accepted: bool,
    pub clauses: Vec<HandshakeClauseVerdict>,
    pub obligations: HandshakeObligations,
    pub taskset_resolution: Vec<TasksetResolutionEntry>,
    pub capability_hash: String,
    /// Binds both documents plus the capability hash, the renderer profile, the
    /// resolved task digests, and the obligations.
    pub agreement_digest: String,
    pub expires_at: String,
    pub clock: HandshakeContainerClock,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl CispoHandshakeVerdict {
    pub fn clause(&self, clause_id: &str) -> Option<&HandshakeClauseVerdict> {
        self.clauses.iter().find(|c| c.clause_id == clause_id)
    }

    /// Mandatory clauses that stop the run: answered rejected or unsupported,
    /// or not answered at all. An unanswered mandatory clause is blocking
    /// because silence is not agreement.
    ///
    /// `joint_episode` selects whether the topology-scoped clauses apply; they
    /// are meaningless for a single-instance run.
    pub fn blocking_clauses(&self, joint_episode: bool) -> Vec<String> {
        let mut blocking = Vec::new();
        for clause_id in cispo_mandatory_clauses() {
            if !joint_episode && CISPO_TOPOLOGY_ONLY_CLAUSES.contains(&clause_id) {
                continue;
            }
            match self.clause(clause_id) {
                None => blocking.push(clause_id.to_string()),
                Some(clause) if clause.verdict.blocks_mandatory() => {
                    blocking.push(clause_id.to_string())
                }
                Some(_) => {}
            }
        }
        blocking
    }

    /// Optional clauses the run must record a fallback for. These do not stop
    /// the run; not recording them is what would.
    pub fn fallback_clauses(&self) -> Vec<String> {
        self.clauses
            .iter()
            .filter(|c| CISPO_OPTIONAL_CLAUSES.contains(&c.clause_id.as_str()))
            .filter(|c| c.verdict.needs_fallback())
            .map(|c| c.clause_id.clone())
            .collect()
    }

    /// Clauses the executor can only satisfy by lowering its own run plan. The
    /// lowered plan is re-handshaked rather than assumed.
    pub fn degraded_clauses(&self) -> Vec<String> {
        self.clauses
            .iter()
            .filter(|c| c.verdict.needs_replan())
            .map(|c| c.clause_id.clone())
            .collect()
    }

    /// Skew beyond the declared tolerance is a rejection, because the horizon
    /// is the instant the reward is read.
    pub fn assert_clock_skew_within(&self, tolerance_seconds: f64) -> Result<()> {
        let skew = self.clock.measured_skew_seconds.abs();
        if !skew.is_finite() || skew > tolerance_seconds {
            return Err(OptimizerError::Container(format!(
                "measured clock skew {skew}s exceeds tolerance {tolerance_seconds}s"
            )));
        }
        Ok(())
    }

    /// Shape, then agreement. Errors when any mandatory clause blocks; the
    /// caller reads `fallback_clauses` for what it must record.
    pub fn validate(&self, request: &CispoHandshakeRequest, joint_episode: bool) -> Result<()> {
        if self.schema_version != CISPO_HANDSHAKE_SCHEMA_VERSION {
            return Err(OptimizerError::Container(format!(
                "handshake verdict schema_version must be {CISPO_HANDSHAKE_SCHEMA_VERSION:?}, got {:?}",
                self.schema_version
            )));
        }
        for (name, value) in [
            ("handshake_id", self.handshake_id.as_str()),
            ("agreement_digest", self.agreement_digest.as_str()),
            ("expires_at", self.expires_at.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(OptimizerError::Container(format!(
                    "handshake verdict must include {name}"
                )));
            }
        }
        if self.capability_hash != request.capability_hash {
            return Err(OptimizerError::Container(format!(
                "handshake verdict capability_hash {:?} does not match the requested {:?}; \
                 the capability document changed under the preflight",
                self.capability_hash, request.capability_hash
            )));
        }
        for clause in &self.clauses {
            cispo_clause_group(&clause.clause_id)?;
        }
        let blocking = self.blocking_clauses(joint_episode);
        if !blocking.is_empty() {
            return Err(OptimizerError::Container(format!(
                "handshake rejected mandatory clauses {blocking:?}; the run stops before session creation"
            )));
        }
        if !self.accepted {
            return Err(OptimizerError::Container(
                "handshake verdict is not accepted".to_string(),
            ));
        }
        if self.obligations.max_concurrency < request.run_plan.group_size {
            return Err(OptimizerError::Container(format!(
                "obligated max_concurrency {} is below the requested group_size {}",
                self.obligations.max_concurrency, request.run_plan.group_size
            )));
        }
        if let Some(horizon) = &self.obligations.horizon {
            horizon.validate()?;
        }
        Ok(())
    }
}

pub fn decode_cispo_handshake_verdict(value: Value) -> Result<CispoHandshakeVerdict> {
    Ok(serde_json::from_value(value)?)
}
