//! One attempt, from submission to sealed evidence and reward.
//!
//! The container does not need to understand group advantages. It only needs to
//! round-trip the correlation fields and execute each requested attempt exactly
//! once, then hand back evidence and a reward bound to it.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::capabilities::HorizonDoc;
use super::{
    CISPO_ATTEMPT_STATES, CISPO_REWARD_RECORD_SCHEMA_VERSION, CISPO_TERMINAL_ATTEMPT_STATES,
    CISPO_TRAINABLE_EPISODE_SCHEMA_VERSION,
};
use crate::container_contract::JsonMap;
use crate::error::{OptimizerError, Result};

/// One attempt. Every correlation field here is opaque to the container.
///
/// The design note writes the joint-episode field as `policy_set_revision`
/// while `rl_identity.RolloutReceipt` and `rl_records.InferenceCall` both call
/// it `policy_set_revision_id`. The Python name wins; the note's spelling is an
/// alias.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CispoRolloutSubmission {
    /// Retrying this key may not create a second logical attempt.
    pub idempotency_key: String,
    pub rollout_id: Option<String>,
    pub handshake_id: String,
    pub agreement_digest: String,
    pub run_id: String,
    pub group_id: String,
    pub sample_index: u32,
    pub seed: i64,
    pub policy_revision: i64,
    pub behavior_fingerprint: String,
    pub policy_config_id: Option<String>,
    #[serde(alias = "policy_set_revision")]
    pub policy_set_revision_id: Option<String>,
    pub agent_instance_id: Option<String>,
    pub team_id: Option<String>,
    pub task_id: String,
    pub taskset_id: Option<String>,
    pub split: Option<String>,
    pub topology_ref: Option<String>,
    pub content_digest: Option<String>,
    /// Probe attempts are non-trainable and can never enter a group or a batch.
    pub probe: bool,
    pub horizon: Option<HorizonDoc>,
    pub metadata: JsonMap,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl CispoRolloutSubmission {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("idempotency_key", self.idempotency_key.as_str()),
            ("handshake_id", self.handshake_id.as_str()),
            ("agreement_digest", self.agreement_digest.as_str()),
            ("run_id", self.run_id.as_str()),
            ("group_id", self.group_id.as_str()),
            ("task_id", self.task_id.as_str()),
            ("behavior_fingerprint", self.behavior_fingerprint.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(OptimizerError::Container(format!(
                    "rollout submission must include {name}"
                )));
            }
        }
        if self.policy_revision < 0 {
            return Err(OptimizerError::Container(
                "policy_revision must be non-negative".to_string(),
            ));
        }
        if let Some(horizon) = &self.horizon {
            horizon.validate()?;
        }
        Ok(())
    }
}

/// The 202 answer: an id, a lease, and the accepted correlation echo.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CispoRolloutAck {
    pub rollout_id: String,
    pub idempotency_key: Option<String>,
    pub state: String,
    pub lease_expires_at: Option<String>,
    /// True when this key was already accepted and the same logical attempt is
    /// being returned. A resubmit must never open a second attempt.
    pub deduplicated: bool,
    pub event_cursor: Option<String>,
    pub metadata: JsonMap,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl CispoRolloutAck {
    pub fn validate_for(&self, submission: &CispoRolloutSubmission) -> Result<()> {
        if self.rollout_id.trim().is_empty() {
            return Err(OptimizerError::Container(
                "rollout submission response must include rollout_id".to_string(),
            ));
        }
        if let Some(echoed) = self.idempotency_key.as_deref() {
            if echoed != submission.idempotency_key {
                return Err(OptimizerError::Container(format!(
                    "rollout ack echoed idempotency_key {echoed:?} for submitted {:?}",
                    submission.idempotency_key
                )));
            }
        }
        if !self.state.is_empty() && !CISPO_ATTEMPT_STATES.contains(&self.state.as_str()) {
            return Err(OptimizerError::Container(format!(
                "unknown attempt state {:?}",
                self.state
            )));
        }
        Ok(())
    }
}

/// Per-instance liveness inside a joint episode. A concurrent real-time
/// topology needs it per stream, because the executor must not serialize the
/// episode into a global turn order to find out who is alive.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RolloutInstanceState {
    pub agent_instance_id: String,
    pub team_id: Option<String>,
    pub state: String,
    pub trainable: bool,
    pub last_seen_at: Option<String>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

/// Attempt state, its lease, and its resumable cursor.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CispoRolloutState {
    pub rollout_id: String,
    pub state: String,
    pub lease_expires_at: Option<String>,
    /// Monotone and resumable. A cursor that goes backwards makes restart
    /// recovery a guess.
    pub event_cursor: Option<String>,
    pub instances: Vec<RolloutInstanceState>,
    pub terminal_status: Option<String>,
    pub trace_digest: Option<String>,
    pub reward_id: Option<String>,
    pub status_detail: Option<String>,
    pub metadata: JsonMap,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl CispoRolloutState {
    pub fn is_terminal(&self) -> bool {
        CISPO_TERMINAL_ATTEMPT_STATES.contains(&self.state.as_str())
    }

    pub fn validate(&self) -> Result<()> {
        if self.rollout_id.trim().is_empty() {
            return Err(OptimizerError::Container(
                "rollout state must include rollout_id".to_string(),
            ));
        }
        if !CISPO_ATTEMPT_STATES.contains(&self.state.as_str()) {
            return Err(OptimizerError::Container(format!(
                "unknown attempt state {:?}",
                self.state
            )));
        }
        if let Some(status) = self.terminal_status.as_deref() {
            if !CISPO_TERMINAL_ATTEMPT_STATES.contains(&status) {
                return Err(OptimizerError::Container(format!(
                    "receipt terminal_status {status:?} is not terminal"
                )));
            }
        }
        // Active work must be recoverable, which means a lease. Queued work has
        // not been leased yet and is discardable by design.
        if !self.is_terminal() && self.state != "queued" && self.lease_expires_at.is_none() {
            return Err(OptimizerError::Container(format!(
                "rollout {} is {} with no lease expiry; active work must be recoverable",
                self.rollout_id, self.state
            )));
        }
        Ok(())
    }
}

/// Sealed evidence, inline or by reference plus digest.
///
/// A bundle too large to inline is stored by reference and the reference must
/// resolve for the retention life of the run: gigabytes of recordings must not
/// be forced through the job store.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CispoTraceReference {
    pub rollout_id: String,
    pub trace_digest: String,
    pub schema_version: String,
    pub inline: Option<Value>,
    pub uri: Option<String>,
    pub media_type: Option<String>,
    pub size_bytes: Option<u64>,
    pub expires_at: Option<String>,
    pub segment_count: Option<u64>,
    pub metadata: JsonMap,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl CispoTraceReference {
    /// Missing or malformed trainable evidence is a terminal evidence failure,
    /// never a zero-reward trajectory. This errors rather than returning an
    /// empty trace for exactly that reason.
    pub fn validate(&self) -> Result<()> {
        if self.rollout_id.trim().is_empty() {
            return Err(OptimizerError::Container(
                "trace reference must include rollout_id".to_string(),
            ));
        }
        if self.trace_digest.trim().is_empty() {
            return Err(OptimizerError::Container(format!(
                "episode {} has no sealed trace digest",
                self.rollout_id
            )));
        }
        if self.schema_version != CISPO_TRAINABLE_EPISODE_SCHEMA_VERSION {
            return Err(OptimizerError::Container(format!(
                "trace schema_version must be {CISPO_TRAINABLE_EPISODE_SCHEMA_VERSION:?}, got {:?}",
                self.schema_version
            )));
        }
        if self.inline.is_none() && self.uri.as_deref().unwrap_or_default().trim().is_empty() {
            return Err(OptimizerError::Container(format!(
                "trace for {} is neither inline nor resolvable by reference",
                self.rollout_id
            )));
        }
        Ok(())
    }

    pub fn is_inline(&self) -> bool {
        self.inline.is_some()
    }
}

/// One team's measure. Absolute and rank are both recorded, so a competitive
/// relation cannot quietly become an absolute one.
///
/// Mirrors `rl_records.RewardChannel`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct RewardChannelDoc {
    pub channel_id: String,
    pub team_id: Option<String>,
    pub measure: f64,
    pub rank: Option<i64>,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl RewardChannelDoc {
    pub fn validate(&self) -> Result<()> {
        if !self.measure.is_finite() {
            return Err(OptimizerError::Container(format!(
                "reward channel {} measure is not finite",
                self.channel_id
            )));
        }
        Ok(())
    }
}

/// When the reward was read, and whether the environment was still moving.
///
/// Mirrors `rl_records.HorizonEvidence`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HorizonEvidenceDoc {
    pub horizon_kind: String,
    pub horizon_value: f64,
    pub scored_at_offset_seconds: f64,
    pub clipped: bool,
    pub quiescence_attested: bool,
    pub settlement_window_seconds: f64,
    pub credited_settlement_seconds: f64,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl HorizonEvidenceDoc {
    pub fn validate(&self) -> Result<()> {
        if self.scored_at_offset_seconds > self.settlement_window_seconds && !self.clipped {
            return Err(OptimizerError::Container(
                "reward was read past the horizon and settlement window without clipping"
                    .to_string(),
            ));
        }
        if !self.quiescence_attested && !self.clipped {
            return Err(OptimizerError::Container(
                "reward has neither a quiescence attestation nor a horizon-clipped snapshot"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Container-authoritative reward, bound to the rollout and the sealed trace
/// digest.
///
/// Mirrors `rl_records.RewardRecord`. Zero stays distinguishable from absent:
/// an absent channel is an error here, not a zero.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CispoRewardReceipt {
    pub reward_id: String,
    pub rollout_id: String,
    pub trace_digest: String,
    pub channels: Vec<RewardChannelDoc>,
    pub optimized_channel: String,
    pub terminal_status: String,
    pub evaluation_plan_id: String,
    pub horizon: Option<HorizonEvidenceDoc>,
    pub metadata: JsonMap,
    pub schema_version: String,
    #[serde(flatten)]
    pub extra: JsonMap,
}

impl CispoRewardReceipt {
    /// `episode_trace_digest` binds the receipt to the evidence it scored. Pass
    /// it whenever the sealed trace is in hand; without it the binding is
    /// unchecked, which is the one thing the reward contract exists to prevent.
    pub fn validate(&self, episode_trace_digest: Option<&str>) -> Result<()> {
        if self.schema_version != CISPO_REWARD_RECORD_SCHEMA_VERSION {
            return Err(OptimizerError::Container(format!(
                "reward receipt schema_version must be {CISPO_REWARD_RECORD_SCHEMA_VERSION:?}, got {:?}",
                self.schema_version
            )));
        }
        if self.reward_id.trim().is_empty() {
            return Err(OptimizerError::Container(
                "reward receipt must include reward_id".to_string(),
            ));
        }
        if self.rollout_id.trim().is_empty() {
            return Err(OptimizerError::Container(format!(
                "reward {} is not bound to a rollout id",
                self.reward_id
            )));
        }
        if self.channels.is_empty() {
            return Err(OptimizerError::Container(format!(
                "reward {} carries no channel; absent is not zero",
                self.reward_id
            )));
        }
        for channel in &self.channels {
            channel.validate()?;
        }
        if self.trace_digest.trim().is_empty() {
            return Err(OptimizerError::Container(format!(
                "reward {} is not bound to a trace digest",
                self.reward_id
            )));
        }
        if let Some(expected) = episode_trace_digest {
            if expected != self.trace_digest {
                return Err(OptimizerError::Container(format!(
                    "reward {} trace digest does not match its episode",
                    self.reward_id
                )));
            }
        }
        if !self
            .channels
            .iter()
            .any(|channel| channel.channel_id == self.optimized_channel)
        {
            return Err(OptimizerError::Container(format!(
                "reward {} optimizes channel {:?} which it does not carry",
                self.reward_id, self.optimized_channel
            )));
        }
        if !CISPO_TERMINAL_ATTEMPT_STATES.contains(&self.terminal_status.as_str()) {
            return Err(OptimizerError::Container(format!(
                "receipt terminal_status {:?} is not terminal",
                self.terminal_status
            )));
        }
        if self.evaluation_plan_id.trim().is_empty() {
            return Err(OptimizerError::Container(format!(
                "reward {} does not name a stable evaluation_plan_id",
                self.reward_id
            )));
        }
        if let Some(horizon) = &self.horizon {
            horizon.validate()?;
        }
        Ok(())
    }

    /// The measure of the named channel, or of the optimized one.
    pub fn value(&self, channel_id: Option<&str>) -> Result<f64> {
        let wanted = channel_id.unwrap_or(self.optimized_channel.as_str());
        self.channels
            .iter()
            .find(|channel| channel.channel_id == wanted)
            .map(|channel| channel.measure)
            .ok_or_else(|| {
                OptimizerError::Container(format!(
                    "reward {} has no channel {wanted:?}",
                    self.reward_id
                ))
            })
    }
}

pub fn decode_cispo_rollout_ack(value: Value) -> Result<CispoRolloutAck> {
    Ok(serde_json::from_value(value)?)
}

pub fn decode_cispo_rollout_state(value: Value) -> Result<CispoRolloutState> {
    let state: CispoRolloutState = serde_json::from_value(value)?;
    state.validate()?;
    Ok(state)
}

pub fn decode_cispo_trace_reference(value: Value) -> Result<CispoTraceReference> {
    let trace: CispoTraceReference = serde_json::from_value(value)?;
    trace.validate()?;
    Ok(trace)
}

pub fn decode_cispo_reward_receipt(
    value: Value,
    episode_trace_digest: Option<&str>,
) -> Result<CispoRewardReceipt> {
    let receipt: CispoRewardReceipt = serde_json::from_value(value)?;
    receipt.validate(episode_trace_digest)?;
    Ok(receipt)
}
