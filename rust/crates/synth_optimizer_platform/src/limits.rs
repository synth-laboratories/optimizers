use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use time::OffsetDateTime;

use crate::cache::stable_value_hash;
use crate::error::{OptimizerError, Result};

pub const RUN_LIMITS_SCHEMA_VERSION: &str = "run_limits.v1";
pub const RUNTIME_EFFECT_ADMISSION_SCHEMA_VERSION: &str = "runtime_effect_admission.v1";
pub const BUDGET_RESERVATION_SCHEMA_VERSION: &str = "budget_reservation.v1";
pub const BUDGET_COMMIT_SCHEMA_VERSION: &str = "budget_commit.v1";
pub const BUDGET_RELEASE_SCHEMA_VERSION: &str = "budget_release.v1";

pub const GEPA_ADAPTIVE_ROLLOUT_CONCURRENCY_ENABLED: bool = true;
pub const GEPA_ADAPTIVE_ROLLOUT_CONCURRENCY_INITIAL: usize = 50;
pub const GEPA_ADAPTIVE_ROLLOUT_CONCURRENCY_MIN: usize = 1;
pub const GEPA_ADAPTIVE_ROLLOUT_CONCURRENCY_MAX: usize = 120;
pub const GEPA_ADAPTIVE_ROLLOUT_CONCURRENCY_INCREASE_STEP: usize = 10;
pub const GEPA_ADAPTIVE_ROLLOUT_CONCURRENCY_DECREASE_STEP: usize = 10;
pub const GEPA_ADAPTIVE_ROLLOUT_CONCURRENCY_INCREASE_AFTER_SUCCESSES: usize = 40;
pub const GEPA_ADAPTIVE_ROLLOUT_CONCURRENCY_OVERLOAD_STATUS_CODES: &[u16] =
    &[408, 425, 429, 500, 502, 503, 504, 529];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunLimitsRecord {
    pub schema_version: String,
    pub run_limits_id: String,
    pub run_id: String,
    #[serde(default)]
    pub max_total_rollouts: Option<u64>,
    #[serde(default)]
    pub max_cost_usd: Option<f64>,
    #[serde(default)]
    pub max_time_seconds: Option<u64>,
    #[serde(default)]
    pub max_prompt_tokens: Option<u64>,
    #[serde(default)]
    pub max_completion_tokens: Option<u64>,
    #[serde(default)]
    pub max_total_tokens: Option<u64>,
    pub hard_limit: bool,
    pub stop_policy: String,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    pub updated_at: String,
}

pub struct RunLimitsInput<'a> {
    pub run_id: &'a str,
    pub max_total_rollouts: Option<u64>,
    pub max_cost_usd: Option<f64>,
    pub max_time_seconds: Option<u64>,
    pub max_prompt_tokens: Option<u64>,
    pub max_completion_tokens: Option<u64>,
    pub max_total_tokens: Option<u64>,
    pub hard_limit: bool,
    pub stop_policy: &'a str,
    pub metadata: Map<String, Value>,
}

pub trait RunLimitPolicy {
    fn max_total_rollouts(&self) -> Option<u64>;
    fn max_cost_usd(&self) -> Option<f64>;
    fn max_time_seconds(&self) -> Option<u64>;
    fn max_prompt_tokens(&self) -> Option<u64>;
    fn max_completion_tokens(&self) -> Option<u64>;
    fn max_total_tokens(&self) -> Option<u64>;
    fn hard_limit(&self) -> bool;
    fn stop_policy(&self) -> &str;

    fn to_run_limits_record(&self, run_id: &str, metadata: Map<String, Value>) -> RunLimitsRecord {
        RunLimitsRecord::from_input(RunLimitsInput {
            run_id,
            max_total_rollouts: self.max_total_rollouts(),
            max_cost_usd: self.max_cost_usd(),
            max_time_seconds: self.max_time_seconds(),
            max_prompt_tokens: self.max_prompt_tokens(),
            max_completion_tokens: self.max_completion_tokens(),
            max_total_tokens: self.max_total_tokens(),
            hard_limit: self.hard_limit(),
            stop_policy: self.stop_policy(),
            metadata,
        })
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BudgetLedgerSnapshot {
    pub run_id: String,
    pub spent_cost_usd: f64,
    pub reserved_cost_usd: f64,
    pub spent_prompt_tokens: u64,
    pub reserved_prompt_tokens: u64,
    pub spent_completion_tokens: u64,
    pub reserved_completion_tokens: u64,
    pub spent_total_tokens: u64,
    pub reserved_total_tokens: u64,
    pub spent_rollouts: u64,
    pub reserved_rollouts: u64,
    pub spent_wall_seconds: u64,
    pub reserved_wall_seconds: u64,
    #[serde(default)]
    pub max_cost_usd: Option<f64>,
    #[serde(default)]
    pub max_prompt_tokens: Option<u64>,
    #[serde(default)]
    pub max_completion_tokens: Option<u64>,
    #[serde(default)]
    pub max_total_tokens: Option<u64>,
    #[serde(default)]
    pub max_total_rollouts: Option<u64>,
    #[serde(default)]
    pub max_time_seconds: Option<u64>,
    #[serde(default)]
    pub remaining_cost_usd: Option<f64>,
    #[serde(default)]
    pub remaining_prompt_tokens: Option<u64>,
    #[serde(default)]
    pub remaining_completion_tokens: Option<u64>,
    #[serde(default)]
    pub remaining_total_tokens: Option<u64>,
    #[serde(default)]
    pub remaining_rollouts: Option<u64>,
    #[serde(default)]
    pub remaining_wall_seconds: Option<u64>,
    pub hard_limit: bool,
    pub status: String,
    pub recorded_at: String,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BudgetLedgerTotals {
    pub cost_usd: f64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub rollouts: u64,
    pub wall_seconds: u64,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct RuntimeEffectBudgetEstimate {
    #[serde(default)]
    pub max_cost_usd: Option<f64>,
    #[serde(default)]
    pub max_prompt_tokens: Option<u64>,
    #[serde(default)]
    pub max_completion_tokens: Option<u64>,
    #[serde(default)]
    pub max_total_tokens: Option<u64>,
    #[serde(default)]
    pub max_rollouts: Option<u64>,
    #[serde(default)]
    pub max_wall_seconds: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct BudgetLimitBreach {
    pub limit: String,
    pub requested: String,
    pub available: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BudgetReleaseRecord {
    pub schema_version: String,
    pub budget_release_id: String,
    pub run_id: String,
    pub runtime_effect_id: String,
    pub budget_reservation_id: String,
    pub release_reason: String,
    pub released_cost_usd: f64,
    pub released_prompt_tokens: u64,
    pub released_completion_tokens: u64,
    pub released_total_tokens: u64,
    pub released_rollouts: u64,
    pub released_wall_seconds: u64,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    pub released_at: String,
}

pub struct BudgetReleaseInput<'a> {
    pub run_id: &'a str,
    pub runtime_effect_id: &'a str,
    pub budget_reservation_id: &'a str,
    pub release_reason: &'a str,
    pub released_cost_usd: f64,
    pub released_prompt_tokens: u64,
    pub released_completion_tokens: u64,
    pub released_total_tokens: u64,
    pub released_rollouts: u64,
    pub released_wall_seconds: u64,
    pub metadata: Map<String, Value>,
}

impl RuntimeEffectBudgetEstimate {
    pub fn requested_budget(self) -> BudgetLedgerTotals {
        let prompt_tokens = self.max_prompt_tokens.unwrap_or(0);
        let completion_tokens = self.max_completion_tokens.unwrap_or(0);
        BudgetLedgerTotals {
            cost_usd: self.max_cost_usd.unwrap_or(0.0),
            prompt_tokens,
            completion_tokens,
            total_tokens: self
                .max_total_tokens
                .unwrap_or_else(|| prompt_tokens.saturating_add(completion_tokens)),
            rollouts: self.max_rollouts.unwrap_or(0),
            wall_seconds: self.max_wall_seconds.unwrap_or(0),
        }
    }

    pub fn validate_for_limits(
        self,
        run_id: &str,
        effect_kind: &str,
        limits: &RunLimitsRecord,
    ) -> Result<()> {
        if limits.max_cost_usd.is_some() {
            require_positive_f64_estimate(run_id, effect_kind, "max_cost_usd", self.max_cost_usd)?;
        }
        if limits.max_prompt_tokens.is_some() {
            require_positive_u64_estimate(
                run_id,
                effect_kind,
                "max_prompt_tokens",
                self.max_prompt_tokens,
            )?;
        }
        if limits.max_completion_tokens.is_some() {
            require_positive_u64_estimate(
                run_id,
                effect_kind,
                "max_completion_tokens",
                self.max_completion_tokens,
            )?;
        }
        if limits.max_total_tokens.is_some() {
            let total = self.max_total_tokens.or_else(|| {
                self.max_prompt_tokens
                    .zip(self.max_completion_tokens)
                    .map(|(a, b)| a.saturating_add(b))
            });
            require_positive_u64_estimate(run_id, effect_kind, "max_total_tokens", total)?;
        }
        if limits.max_total_rollouts.is_some() && effect_kind == "container_rollout" {
            require_positive_u64_estimate(
                run_id,
                effect_kind,
                "max_total_rollouts",
                self.max_rollouts,
            )?;
        }
        if limits.max_time_seconds.is_some() {
            require_positive_u64_estimate(
                run_id,
                effect_kind,
                "max_time_seconds",
                self.max_wall_seconds,
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeEffectAdmissionRecord {
    pub schema_version: String,
    pub admission_id: String,
    pub run_id: String,
    pub runtime_effect_id: String,
    pub effect_kind: String,
    pub lane: String,
    pub subject_type: String,
    pub subject_id: String,
    pub idempotency_key: String,
    pub status: String,
    #[serde(default)]
    pub rejection_reason: Option<String>,
    #[serde(default)]
    pub max_cost_usd: Option<f64>,
    #[serde(default)]
    pub max_prompt_tokens: Option<u64>,
    #[serde(default)]
    pub max_completion_tokens: Option<u64>,
    #[serde(default)]
    pub max_total_tokens: Option<u64>,
    #[serde(default)]
    pub max_rollouts: Option<u64>,
    #[serde(default)]
    pub max_wall_seconds: Option<u64>,
    pub ledger: BudgetLedgerSnapshot,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    pub checked_at: String,
}

pub struct RuntimeEffectAdmissionInput<'a> {
    pub run_id: &'a str,
    pub runtime_effect_id: &'a str,
    pub effect_kind: &'a str,
    pub lane: &'a str,
    pub subject_type: &'a str,
    pub subject_id: &'a str,
    pub idempotency_key: &'a str,
    pub status: &'a str,
    pub rejection_reason: Option<String>,
    pub max_cost_usd: Option<f64>,
    pub max_prompt_tokens: Option<u64>,
    pub max_completion_tokens: Option<u64>,
    pub max_total_tokens: Option<u64>,
    pub max_rollouts: Option<u64>,
    pub max_wall_seconds: Option<u64>,
    pub ledger: BudgetLedgerSnapshot,
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BudgetReservationRecord {
    pub schema_version: String,
    pub budget_reservation_id: String,
    pub run_id: String,
    pub runtime_effect_id: String,
    pub status: String,
    #[serde(default)]
    pub max_cost_usd: Option<f64>,
    #[serde(default)]
    pub max_prompt_tokens: Option<u64>,
    #[serde(default)]
    pub max_completion_tokens: Option<u64>,
    #[serde(default)]
    pub max_total_tokens: Option<u64>,
    #[serde(default)]
    pub max_rollouts: Option<u64>,
    #[serde(default)]
    pub max_wall_seconds: Option<u64>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    pub reserved_at: String,
    pub updated_at: String,
}

pub struct BudgetReservationInput<'a> {
    pub run_id: &'a str,
    pub runtime_effect_id: &'a str,
    pub status: &'a str,
    pub max_cost_usd: Option<f64>,
    pub max_prompt_tokens: Option<u64>,
    pub max_completion_tokens: Option<u64>,
    pub max_total_tokens: Option<u64>,
    pub max_rollouts: Option<u64>,
    pub max_wall_seconds: Option<u64>,
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BudgetCommitRecord {
    pub schema_version: String,
    pub budget_commit_id: String,
    pub run_id: String,
    pub runtime_effect_id: String,
    pub budget_reservation_id: String,
    pub cost_usd: f64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub rollout_count: u64,
    pub wall_seconds: u64,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    pub committed_at: String,
}

pub struct BudgetCommitInput<'a> {
    pub run_id: &'a str,
    pub runtime_effect_id: &'a str,
    pub budget_reservation_id: &'a str,
    pub cost_usd: f64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub rollout_count: u64,
    pub wall_seconds: u64,
    pub metadata: Map<String, Value>,
}

impl RunLimitsRecord {
    pub fn from_input(input: RunLimitsInput<'_>) -> Self {
        Self {
            schema_version: RUN_LIMITS_SCHEMA_VERSION.to_string(),
            run_limits_id: format!("run_limits_{}", input.run_id),
            run_id: input.run_id.to_string(),
            max_total_rollouts: input.max_total_rollouts,
            max_cost_usd: input.max_cost_usd.filter(|value| *value > 0.0),
            max_time_seconds: input.max_time_seconds,
            max_prompt_tokens: input.max_prompt_tokens,
            max_completion_tokens: input.max_completion_tokens,
            max_total_tokens: input.max_total_tokens,
            hard_limit: input.hard_limit,
            stop_policy: input.stop_policy.to_string(),
            metadata: input.metadata,
            updated_at: now_rfc3339(),
        }
    }
}

impl BudgetLedgerSnapshot {
    pub fn from_totals(
        run_id: &str,
        limits: &RunLimitsRecord,
        spent: BudgetLedgerTotals,
        reserved: BudgetLedgerTotals,
    ) -> Self {
        let status = budget_ledger_status(limits, &spent, &reserved);
        let remaining_cost_usd = limits
            .max_cost_usd
            .map(|max| (max - spent.cost_usd - reserved.cost_usd).max(0.0));
        let remaining_prompt_tokens = remaining_u64(
            limits.max_prompt_tokens,
            spent.prompt_tokens,
            reserved.prompt_tokens,
        );
        let remaining_completion_tokens = remaining_u64(
            limits.max_completion_tokens,
            spent.completion_tokens,
            reserved.completion_tokens,
        );
        let remaining_total_tokens = remaining_u64(
            limits.max_total_tokens,
            spent.total_tokens,
            reserved.total_tokens,
        );
        let remaining_rollouts =
            remaining_u64(limits.max_total_rollouts, spent.rollouts, reserved.rollouts);
        let remaining_wall_seconds = remaining_u64(
            limits.max_time_seconds,
            spent.wall_seconds,
            reserved.wall_seconds,
        );
        Self {
            run_id: run_id.to_string(),
            spent_cost_usd: spent.cost_usd,
            reserved_cost_usd: reserved.cost_usd,
            spent_prompt_tokens: spent.prompt_tokens,
            reserved_prompt_tokens: reserved.prompt_tokens,
            spent_completion_tokens: spent.completion_tokens,
            reserved_completion_tokens: reserved.completion_tokens,
            spent_total_tokens: spent.total_tokens,
            reserved_total_tokens: reserved.total_tokens,
            spent_rollouts: spent.rollouts,
            reserved_rollouts: reserved.rollouts,
            spent_wall_seconds: spent.wall_seconds,
            reserved_wall_seconds: reserved.wall_seconds,
            max_cost_usd: limits.max_cost_usd,
            max_prompt_tokens: limits.max_prompt_tokens,
            max_completion_tokens: limits.max_completion_tokens,
            max_total_tokens: limits.max_total_tokens,
            max_total_rollouts: limits.max_total_rollouts,
            max_time_seconds: limits.max_time_seconds,
            remaining_cost_usd,
            remaining_prompt_tokens,
            remaining_completion_tokens,
            remaining_total_tokens,
            remaining_rollouts,
            remaining_wall_seconds,
            hard_limit: limits.hard_limit,
            status,
            recorded_at: now_rfc3339(),
        }
    }

    pub fn breach_for_request(&self, request: BudgetLedgerTotals) -> Option<BudgetLimitBreach> {
        f64_limit_breach(
            "max_cost_usd",
            self.max_cost_usd,
            self.spent_cost_usd + self.reserved_cost_usd,
            request.cost_usd,
        )
        .or_else(|| {
            u64_limit_breach(
                "max_prompt_tokens",
                self.max_prompt_tokens,
                self.spent_prompt_tokens + self.reserved_prompt_tokens,
                request.prompt_tokens,
            )
        })
        .or_else(|| {
            u64_limit_breach(
                "max_completion_tokens",
                self.max_completion_tokens,
                self.spent_completion_tokens + self.reserved_completion_tokens,
                request.completion_tokens,
            )
        })
        .or_else(|| {
            u64_limit_breach(
                "max_total_tokens",
                self.max_total_tokens,
                self.spent_total_tokens + self.reserved_total_tokens,
                request.total_tokens,
            )
        })
        .or_else(|| {
            u64_limit_breach(
                "max_total_rollouts",
                self.max_total_rollouts,
                self.spent_rollouts + self.reserved_rollouts,
                request.rollouts,
            )
        })
        .or_else(|| {
            u64_limit_breach(
                "max_time_seconds",
                self.max_time_seconds,
                self.spent_wall_seconds + self.reserved_wall_seconds,
                request.wall_seconds,
            )
        })
    }

    pub fn exceeded_limit(&self) -> Option<BudgetLimitBreach> {
        f64_limit_exceeded(
            "max_cost_usd",
            self.max_cost_usd,
            self.spent_cost_usd + self.reserved_cost_usd,
        )
        .or_else(|| {
            u64_limit_exceeded(
                "max_prompt_tokens",
                self.max_prompt_tokens,
                self.spent_prompt_tokens + self.reserved_prompt_tokens,
            )
        })
        .or_else(|| {
            u64_limit_exceeded(
                "max_completion_tokens",
                self.max_completion_tokens,
                self.spent_completion_tokens + self.reserved_completion_tokens,
            )
        })
        .or_else(|| {
            u64_limit_exceeded(
                "max_total_tokens",
                self.max_total_tokens,
                self.spent_total_tokens + self.reserved_total_tokens,
            )
        })
        .or_else(|| {
            u64_limit_exceeded(
                "max_total_rollouts",
                self.max_total_rollouts,
                self.spent_rollouts + self.reserved_rollouts,
            )
        })
        .or_else(|| {
            u64_limit_exceeded(
                "max_time_seconds",
                self.max_time_seconds,
                self.spent_wall_seconds + self.reserved_wall_seconds,
            )
        })
    }

    pub fn remaining_cost_usd(&self) -> Option<f64> {
        self.remaining_cost_usd
    }

    pub fn remaining_prompt_tokens(&self) -> Option<u64> {
        self.remaining_prompt_tokens
    }

    pub fn remaining_completion_tokens(&self) -> Option<u64> {
        self.remaining_completion_tokens
    }

    pub fn remaining_total_tokens(&self) -> Option<u64> {
        self.remaining_total_tokens
    }

    pub fn remaining_rollouts(&self) -> Option<u64> {
        self.remaining_rollouts
    }

    pub fn remaining_wall_seconds(&self) -> Option<u64> {
        self.remaining_wall_seconds
    }

    pub fn active_reservations(&self) -> BudgetLedgerTotals {
        BudgetLedgerTotals {
            cost_usd: self.reserved_cost_usd,
            prompt_tokens: self.reserved_prompt_tokens,
            completion_tokens: self.reserved_completion_tokens,
            total_tokens: self.reserved_total_tokens,
            rollouts: self.reserved_rollouts,
            wall_seconds: self.reserved_wall_seconds,
        }
    }

    pub fn is_exhausted(&self) -> bool {
        self.status == "exhausted"
    }
}

impl RuntimeEffectAdmissionRecord {
    pub fn from_input(input: RuntimeEffectAdmissionInput<'_>) -> Self {
        let identity = json!({
            "schema_version": RUNTIME_EFFECT_ADMISSION_SCHEMA_VERSION,
            "run_id": input.run_id,
            "runtime_effect_id": input.runtime_effect_id,
            "status": input.status,
            "idempotency_key": input.idempotency_key,
        });
        Self {
            schema_version: RUNTIME_EFFECT_ADMISSION_SCHEMA_VERSION.to_string(),
            admission_id: prefixed_hash_id("effect_admission", &identity),
            run_id: input.run_id.to_string(),
            runtime_effect_id: input.runtime_effect_id.to_string(),
            effect_kind: input.effect_kind.to_string(),
            lane: input.lane.to_string(),
            subject_type: input.subject_type.to_string(),
            subject_id: input.subject_id.to_string(),
            idempotency_key: input.idempotency_key.to_string(),
            status: input.status.to_string(),
            rejection_reason: input.rejection_reason,
            max_cost_usd: input.max_cost_usd.filter(|value| *value > 0.0),
            max_prompt_tokens: input.max_prompt_tokens,
            max_completion_tokens: input.max_completion_tokens,
            max_total_tokens: input.max_total_tokens,
            max_rollouts: input.max_rollouts,
            max_wall_seconds: input.max_wall_seconds,
            ledger: input.ledger,
            metadata: input.metadata,
            checked_at: now_rfc3339(),
        }
    }
}

impl BudgetReservationRecord {
    pub fn from_input(input: BudgetReservationInput<'_>) -> Self {
        let identity = json!({
            "run_id": input.run_id,
            "runtime_effect_id": input.runtime_effect_id,
        });
        let now = now_rfc3339();
        Self {
            schema_version: BUDGET_RESERVATION_SCHEMA_VERSION.to_string(),
            budget_reservation_id: prefixed_hash_id("budget_reservation", &identity),
            run_id: input.run_id.to_string(),
            runtime_effect_id: input.runtime_effect_id.to_string(),
            status: input.status.to_string(),
            max_cost_usd: input.max_cost_usd.filter(|value| *value > 0.0),
            max_prompt_tokens: input.max_prompt_tokens,
            max_completion_tokens: input.max_completion_tokens,
            max_total_tokens: input.max_total_tokens,
            max_rollouts: input.max_rollouts,
            max_wall_seconds: input.max_wall_seconds,
            metadata: input.metadata,
            reserved_at: now.clone(),
            updated_at: now,
        }
    }

    pub fn reserved_budget(&self) -> BudgetLedgerTotals {
        BudgetLedgerTotals {
            cost_usd: self.max_cost_usd.unwrap_or(0.0),
            prompt_tokens: self.max_prompt_tokens.unwrap_or(0),
            completion_tokens: self.max_completion_tokens.unwrap_or(0),
            total_tokens: self.max_total_tokens.unwrap_or_else(|| {
                self.max_prompt_tokens
                    .unwrap_or(0)
                    .saturating_add(self.max_completion_tokens.unwrap_or(0))
            }),
            rollouts: self.max_rollouts.unwrap_or(0),
            wall_seconds: self.max_wall_seconds.unwrap_or(0),
        }
    }
}

impl BudgetCommitRecord {
    pub fn from_input(input: BudgetCommitInput<'_>) -> Self {
        let identity = json!({
            "run_id": input.run_id,
            "runtime_effect_id": input.runtime_effect_id,
            "budget_reservation_id": input.budget_reservation_id,
        });
        Self {
            schema_version: BUDGET_COMMIT_SCHEMA_VERSION.to_string(),
            budget_commit_id: prefixed_hash_id("budget_commit", &identity),
            run_id: input.run_id.to_string(),
            runtime_effect_id: input.runtime_effect_id.to_string(),
            budget_reservation_id: input.budget_reservation_id.to_string(),
            cost_usd: input.cost_usd,
            prompt_tokens: input.prompt_tokens,
            completion_tokens: input.completion_tokens,
            total_tokens: input.total_tokens,
            rollout_count: input.rollout_count,
            wall_seconds: input.wall_seconds,
            metadata: input.metadata,
            committed_at: now_rfc3339(),
        }
    }

    pub fn committed_budget(&self) -> BudgetLedgerTotals {
        BudgetLedgerTotals {
            cost_usd: self.cost_usd,
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
            rollouts: self.rollout_count,
            wall_seconds: self.wall_seconds,
        }
    }
}

impl BudgetReleaseRecord {
    pub fn from_input(input: BudgetReleaseInput<'_>) -> Self {
        let identity = json!({
            "run_id": input.run_id,
            "runtime_effect_id": input.runtime_effect_id,
            "budget_reservation_id": input.budget_reservation_id,
        });
        Self {
            schema_version: BUDGET_RELEASE_SCHEMA_VERSION.to_string(),
            budget_release_id: prefixed_hash_id("budget_release", &identity),
            run_id: input.run_id.to_string(),
            runtime_effect_id: input.runtime_effect_id.to_string(),
            budget_reservation_id: input.budget_reservation_id.to_string(),
            release_reason: input.release_reason.to_string(),
            released_cost_usd: input.released_cost_usd.max(0.0),
            released_prompt_tokens: input.released_prompt_tokens,
            released_completion_tokens: input.released_completion_tokens,
            released_total_tokens: input.released_total_tokens,
            released_rollouts: input.released_rollouts,
            released_wall_seconds: input.released_wall_seconds,
            metadata: input.metadata,
            released_at: now_rfc3339(),
        }
    }
}

fn budget_ledger_status(
    limits: &RunLimitsRecord,
    spent: &BudgetLedgerTotals,
    reserved: &BudgetLedgerTotals,
) -> String {
    let exhausted = limits
        .max_total_rollouts
        .map(|max| spent.rollouts + reserved.rollouts >= max)
        .unwrap_or(false)
        || limits
            .max_cost_usd
            .map(|max| spent.cost_usd + reserved.cost_usd >= max)
            .unwrap_or(false)
        || limits
            .max_total_tokens
            .map(|max| spent.total_tokens + reserved.total_tokens >= max)
            .unwrap_or(false)
        || limits
            .max_prompt_tokens
            .map(|max| spent.prompt_tokens + reserved.prompt_tokens >= max)
            .unwrap_or(false)
        || limits
            .max_completion_tokens
            .map(|max| spent.completion_tokens + reserved.completion_tokens >= max)
            .unwrap_or(false)
        || limits
            .max_time_seconds
            .map(|max| spent.wall_seconds + reserved.wall_seconds >= max)
            .unwrap_or(false);
    if exhausted {
        "exhausted".to_string()
    } else {
        "within_limits".to_string()
    }
}

fn u64_limit_breach(
    limit: &str,
    max: Option<u64>,
    current: u64,
    requested: u64,
) -> Option<BudgetLimitBreach> {
    let max = max?;
    let available = max.saturating_sub(current);
    if current >= max || requested > available {
        Some(BudgetLimitBreach {
            limit: limit.to_string(),
            requested: requested.to_string(),
            available: available.to_string(),
        })
    } else {
        None
    }
}

fn remaining_u64(max: Option<u64>, spent: u64, reserved: u64) -> Option<u64> {
    max.map(|limit| limit.saturating_sub(spent.saturating_add(reserved)))
}

fn f64_limit_breach(
    limit: &str,
    max: Option<f64>,
    current: f64,
    requested: f64,
) -> Option<BudgetLimitBreach> {
    let max = max?;
    let available = (max - current).max(0.0);
    if current >= max || requested > available {
        Some(BudgetLimitBreach {
            limit: limit.to_string(),
            requested: format!("{requested:.6}"),
            available: format!("{available:.6}"),
        })
    } else {
        None
    }
}

fn u64_limit_exceeded(limit: &str, max: Option<u64>, current: u64) -> Option<BudgetLimitBreach> {
    let max = max?;
    if current > max {
        Some(BudgetLimitBreach {
            limit: limit.to_string(),
            requested: current.to_string(),
            available: max.to_string(),
        })
    } else {
        None
    }
}

fn f64_limit_exceeded(limit: &str, max: Option<f64>, current: f64) -> Option<BudgetLimitBreach> {
    let max = max?;
    if current > max {
        Some(BudgetLimitBreach {
            limit: limit.to_string(),
            requested: format!("{current:.6}"),
            available: format!("{max:.6}"),
        })
    } else {
        None
    }
}

fn require_positive_u64_estimate(
    run_id: &str,
    effect_kind: &str,
    limit: &str,
    value: Option<u64>,
) -> Result<()> {
    if value.unwrap_or(0) == 0 {
        return Err(OptimizerError::Config(format!(
            "run_id={run_id} effect_kind={effect_kind} requires positive budget estimate for {limit}"
        )));
    }
    Ok(())
}

fn require_positive_f64_estimate(
    run_id: &str,
    effect_kind: &str,
    limit: &str,
    value: Option<f64>,
) -> Result<()> {
    if value.unwrap_or(0.0) <= 0.0 {
        return Err(OptimizerError::Config(format!(
            "run_id={run_id} effect_kind={effect_kind} requires positive budget estimate for {limit}"
        )));
    }
    Ok(())
}

fn prefixed_hash_id(prefix: &str, value: &Value) -> String {
    let hash = stable_value_hash(value);
    format!("{prefix}_{}", &hash[..16])
}

fn now_rfc3339() -> String {
    OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}
