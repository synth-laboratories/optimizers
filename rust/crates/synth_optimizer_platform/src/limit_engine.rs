use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use time::OffsetDateTime;

use crate::cache::stable_value_hash;
use crate::limits::{
    BudgetCommitRecord, BudgetLedgerSnapshot, BudgetReservationRecord, RunLimitsRecord,
    RuntimeEffectAdmissionRecord,
};
use crate::runtime_records::RunPhaseTimingRecord;

pub const LIMIT_ENGINE_SCHEMA_VERSION: &str = "limit_engine.v1";
const MIN_FORECAST_INTERVAL_SECONDS: f64 = 1.0;
const MAX_BURST_TO_ELAPSED_RATE_RATIO: f64 = 1.5;
const FORECAST_INTERVAL_LOW_MULTIPLIER: f64 = 0.75;
const FORECAST_INTERVAL_HIGH_MULTIPLIER: f64 = 1.5;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum LimitKind {
    CostUsd,
    TotalRollouts,
    TrainRollouts,
    HeldoutRollouts,
    WallSeconds,
    PromptTokens,
    CompletionTokens,
    TotalTokens,
    Candidates,
    Generations,
    Custom(String),
}

impl LimitKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::CostUsd => "cost_usd",
            Self::TotalRollouts => "total_rollouts",
            Self::TrainRollouts => "train_rollouts",
            Self::HeldoutRollouts => "heldout_rollouts",
            Self::WallSeconds => "wall_seconds",
            Self::PromptTokens => "prompt_tokens",
            Self::CompletionTokens => "completion_tokens",
            Self::TotalTokens => "total_tokens",
            Self::Candidates => "candidates",
            Self::Generations => "generations",
            Self::Custom(value) => value.as_str(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LimitDefinition {
    pub schema_version: String,
    pub limit_id: String,
    pub run_id: String,
    pub kind: LimitKind,
    pub scope: String,
    pub max_value: f64,
    pub hard: bool,
    pub stop_policy: String,
    pub source: String,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LimitObservation {
    pub run_id: String,
    pub limit_id: String,
    pub timestamp: String,
    pub spent: f64,
    pub reserved: f64,
    pub source_kind: String,
    pub source_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LimitProgressEvent {
    pub schema_version: String,
    pub event_id: String,
    pub run_id: String,
    pub limit_id: String,
    pub timestamp: String,
    pub spent: f64,
    pub reserved: f64,
    pub remaining: f64,
    pub utilization: f64,
    pub delta_spent: f64,
    pub delta_reserved: f64,
    pub source_kind: String,
    pub source_id: String,
}

/// Forecast confidence band. A typed, closed set so the interval multipliers
/// match exhaustively — an unhandled band is a compile error, not a silent
/// zero-width interval. (`model` stays a free-form forecaster label; it is
/// display/identity only and never gates behavior.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForecastConfidence {
    High,
    Medium,
    Low,
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LimitForecast {
    pub schema_version: String,
    pub forecast_id: String,
    pub run_id: String,
    pub limit_id: String,
    pub model: String,
    #[serde(default)]
    pub predicted_crossing_at: Option<String>,
    #[serde(default)]
    pub seconds_to_limit: Option<f64>,
    #[serde(default)]
    pub seconds_to_limit_low: Option<f64>,
    #[serde(default)]
    pub seconds_to_limit_high: Option<f64>,
    #[serde(default)]
    pub predicted_crossing_at_low: Option<String>,
    #[serde(default)]
    pub predicted_crossing_at_high: Option<String>,
    #[serde(default)]
    pub rate_per_second: Option<f64>,
    pub confidence: ForecastConfidence,
    pub sample_count: u64,
    pub updated_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LimitStatus {
    pub definition: LimitDefinition,
    pub spent: f64,
    pub reserved: f64,
    pub remaining: f64,
    pub utilization: f64,
    pub forecast: LimitForecast,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LimitSnapshot {
    pub schema_version: String,
    pub run_id: String,
    pub generated_at: String,
    #[serde(default)]
    pub nearest_limit: Option<LimitForecast>,
    pub limits: Vec<LimitStatus>,
    pub events: Vec<LimitProgressEvent>,
}

#[derive(Clone, Debug)]
pub struct LimitEngineInput {
    pub run_id: String,
    pub definitions: Vec<LimitDefinition>,
    pub observations: Vec<LimitObservation>,
    pub generated_at: Option<String>,
}

pub struct LimitEngine;

impl LimitEngine {
    pub fn snapshot(input: LimitEngineInput) -> LimitSnapshot {
        let generated_at = input.generated_at.unwrap_or_else(now_rfc3339);
        let mut definitions = input.definitions;
        definitions.sort_by(|left, right| left.limit_id.cmp(&right.limit_id));
        let definition_ids = definitions
            .iter()
            .map(|definition| definition.limit_id.clone())
            .collect::<BTreeSet<_>>();
        let mut observations = input
            .observations
            .into_iter()
            .filter(|observation| definition_ids.contains(&observation.limit_id))
            .collect::<Vec<_>>();
        observations.sort_by(compare_observations);
        let events = progress_events(&definitions, &observations);
        let mut statuses = Vec::with_capacity(definitions.len());
        for definition in definitions {
            let current = latest_observation_for(&definition.limit_id, &observations)
                .map(|observation| (observation.spent, observation.reserved))
                .unwrap_or((0.0, 0.0));
            let remaining = remaining(definition.max_value, current.0, current.1);
            let utilization = utilization(definition.max_value, current.0, current.1);
            let event_series = events
                .iter()
                .filter(|event| event.limit_id == definition.limit_id)
                .cloned()
                .collect::<Vec<_>>();
            let forecast = forecast_limit(
                &definition,
                current.0,
                remaining,
                &event_series,
                &generated_at,
            );
            statuses.push(LimitStatus {
                definition,
                spent: current.0,
                reserved: current.1,
                remaining,
                utilization,
                forecast,
            });
        }
        let nearest_limit = statuses
            .iter()
            .filter_map(|status| {
                status
                    .forecast
                    .seconds_to_limit
                    .map(|seconds| (seconds, &status.forecast))
            })
            .min_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal))
            .map(|(_, forecast)| forecast.clone());
        LimitSnapshot {
            schema_version: LIMIT_ENGINE_SCHEMA_VERSION.to_string(),
            run_id: input.run_id,
            generated_at,
            nearest_limit,
            limits: statuses,
            events,
        }
    }
}

pub fn budget_limit_snapshot(
    run_id: &str,
    limits: &RunLimitsRecord,
    ledger: &BudgetLedgerSnapshot,
    reservations: &[BudgetReservationRecord],
    commits: &[BudgetCommitRecord],
    admissions: &[RuntimeEffectAdmissionRecord],
    timings: &[RunPhaseTimingRecord],
    generated_at: Option<String>,
) -> LimitSnapshot {
    LimitEngine::snapshot(budget_limit_engine_input(
        run_id,
        limits,
        ledger,
        reservations,
        commits,
        admissions,
        timings,
        generated_at,
    ))
}

pub fn budget_limit_engine_input(
    run_id: &str,
    limits: &RunLimitsRecord,
    ledger: &BudgetLedgerSnapshot,
    reservations: &[BudgetReservationRecord],
    commits: &[BudgetCommitRecord],
    admissions: &[RuntimeEffectAdmissionRecord],
    timings: &[RunPhaseTimingRecord],
    generated_at: Option<String>,
) -> LimitEngineInput {
    let definitions = budget_limit_definitions(run_id, limits);
    let observations = budget_limit_observations(
        run_id,
        ledger,
        &definitions,
        reservations,
        commits,
        admissions,
        timings,
        generated_at.as_deref(),
    );
    LimitEngineInput {
        run_id: run_id.to_string(),
        definitions,
        observations,
        generated_at,
    }
}

fn budget_limit_definitions(run_id: &str, limits: &RunLimitsRecord) -> Vec<LimitDefinition> {
    let mut definitions = Vec::new();
    push_definition(
        &mut definitions,
        run_id,
        LimitKind::CostUsd,
        limits.max_cost_usd,
        limits,
    );
    push_definition(
        &mut definitions,
        run_id,
        LimitKind::PromptTokens,
        limits.max_prompt_tokens.map(|value| value as f64),
        limits,
    );
    push_definition(
        &mut definitions,
        run_id,
        LimitKind::CompletionTokens,
        limits.max_completion_tokens.map(|value| value as f64),
        limits,
    );
    push_definition(
        &mut definitions,
        run_id,
        LimitKind::TotalTokens,
        limits.max_total_tokens.map(|value| value as f64),
        limits,
    );
    push_definition(
        &mut definitions,
        run_id,
        LimitKind::TotalRollouts,
        limits.max_total_rollouts.map(|value| value as f64),
        limits,
    );
    push_definition(
        &mut definitions,
        run_id,
        LimitKind::WallSeconds,
        limits.max_time_seconds.map(|value| value as f64),
        limits,
    );
    definitions
}

fn push_definition(
    definitions: &mut Vec<LimitDefinition>,
    run_id: &str,
    kind: LimitKind,
    max_value: Option<f64>,
    limits: &RunLimitsRecord,
) {
    let Some(max_value) = max_value.filter(|value| value.is_finite() && *value > 0.0) else {
        return;
    };
    let limit_id = format!("{}:{}", run_id, kind.as_str());
    definitions.push(LimitDefinition {
        schema_version: LIMIT_ENGINE_SCHEMA_VERSION.to_string(),
        limit_id,
        run_id: run_id.to_string(),
        kind,
        scope: "run".to_string(),
        max_value,
        hard: limits.hard_limit,
        stop_policy: limits.stop_policy.clone(),
        source: "run_limits".to_string(),
        metadata: limits.metadata.clone(),
    });
}

fn budget_limit_observations(
    run_id: &str,
    ledger: &BudgetLedgerSnapshot,
    definitions: &[LimitDefinition],
    reservations: &[BudgetReservationRecord],
    commits: &[BudgetCommitRecord],
    admissions: &[RuntimeEffectAdmissionRecord],
    timings: &[RunPhaseTimingRecord],
    generated_at: Option<&str>,
) -> Vec<LimitObservation> {
    let mut observations = Vec::new();
    for admission in admissions {
        for definition in definitions {
            let (mut spent, reserved) = ledger_values_for_kind(&definition.kind, &admission.ledger);
            if matches!(&definition.kind, LimitKind::WallSeconds) {
                spent = wall_seconds_from_timings_until(timings, &admission.checked_at);
            }
            observations.push(LimitObservation {
                run_id: run_id.to_string(),
                limit_id: definition.limit_id.clone(),
                timestamp: admission.checked_at.clone(),
                spent,
                reserved,
                source_kind: "runtime_effect_admission".to_string(),
                source_id: admission.admission_id.clone(),
            });
        }
    }
    append_commit_observations(run_id, definitions, commits, &mut observations);
    append_reservation_observations(run_id, definitions, reservations, &mut observations);
    append_timing_observations(run_id, definitions, timings, &mut observations);
    let timestamp = generated_at.map(str::to_string).unwrap_or_else(now_rfc3339);
    for definition in definitions {
        let (mut spent, reserved) = ledger_values_for_kind(&definition.kind, ledger);
        if matches!(&definition.kind, LimitKind::WallSeconds) {
            spent = wall_seconds_from_timings_until(timings, &timestamp);
        }
        observations.push(LimitObservation {
            run_id: run_id.to_string(),
            limit_id: definition.limit_id.clone(),
            timestamp: timestamp.clone(),
            spent,
            reserved,
            source_kind: "budget_ledger_snapshot".to_string(),
            source_id: format!("ledger:{run_id}:{}", definition.kind.as_str()),
        });
    }
    observations
}

fn append_commit_observations(
    run_id: &str,
    definitions: &[LimitDefinition],
    commits: &[BudgetCommitRecord],
    observations: &mut Vec<LimitObservation>,
) {
    let mut sorted = commits.to_vec();
    sorted.sort_by(|left, right| {
        left.committed_at
            .cmp(&right.committed_at)
            .then_with(|| left.budget_commit_id.cmp(&right.budget_commit_id))
    });
    let mut cost_usd = 0.0;
    let mut prompt_tokens = 0.0;
    let mut completion_tokens = 0.0;
    let mut total_tokens = 0.0;
    let mut rollouts = 0.0;
    for commit in sorted {
        cost_usd += commit.cost_usd;
        prompt_tokens += commit.prompt_tokens as f64;
        completion_tokens += commit.completion_tokens as f64;
        total_tokens += commit.total_tokens as f64;
        rollouts += commit.rollout_count as f64;
        for definition in definitions {
            let spent = match &definition.kind {
                LimitKind::CostUsd => cost_usd,
                LimitKind::PromptTokens => prompt_tokens,
                LimitKind::CompletionTokens => completion_tokens,
                LimitKind::TotalTokens => total_tokens,
                LimitKind::TotalRollouts => rollouts,
                LimitKind::WallSeconds => continue,
                _ => continue,
            };
            observations.push(LimitObservation {
                run_id: run_id.to_string(),
                limit_id: definition.limit_id.clone(),
                timestamp: commit.committed_at.clone(),
                spent,
                reserved: 0.0,
                source_kind: "budget_commit".to_string(),
                source_id: commit.budget_commit_id.clone(),
            });
        }
    }
}

fn append_reservation_observations(
    run_id: &str,
    definitions: &[LimitDefinition],
    reservations: &[BudgetReservationRecord],
    observations: &mut Vec<LimitObservation>,
) {
    let mut active = BTreeMap::<String, f64>::new();
    for reservation in reservations {
        if !matches!(
            reservation.status.as_str(),
            "reserved" | "planned" | "active" | "leased" | "running"
        ) {
            continue;
        }
        for definition in definitions {
            let value = reservation_value_for_kind(&definition.kind, reservation);
            if value <= 0.0 {
                continue;
            }
            *active.entry(definition.limit_id.clone()).or_insert(0.0) += value;
            observations.push(LimitObservation {
                run_id: run_id.to_string(),
                limit_id: definition.limit_id.clone(),
                timestamp: reservation.updated_at.clone(),
                spent: 0.0,
                reserved: active.get(&definition.limit_id).copied().unwrap_or(0.0),
                source_kind: "budget_reservation".to_string(),
                source_id: reservation.budget_reservation_id.clone(),
            });
        }
    }
}

fn append_timing_observations(
    run_id: &str,
    definitions: &[LimitDefinition],
    timings: &[RunPhaseTimingRecord],
    observations: &mut Vec<LimitObservation>,
) {
    if !definitions
        .iter()
        .any(|definition| matches!(&definition.kind, LimitKind::WallSeconds))
    {
        return;
    }
    let Some(definition) = definitions
        .iter()
        .find(|definition| matches!(&definition.kind, LimitKind::WallSeconds))
    else {
        return;
    };
    let mut sorted = timings.to_vec();
    sorted.sort_by(|left, right| {
        left.started_at
            .cmp(&right.started_at)
            .then_with(|| left.timing_id.cmp(&right.timing_id))
    });
    let mut wall_seconds = 0.0;
    for timing in sorted {
        let Some(seconds) = timing
            .wall_seconds
            .filter(|value| value.is_finite() && *value > 0.0)
        else {
            continue;
        };
        wall_seconds += seconds;
        observations.push(LimitObservation {
            run_id: run_id.to_string(),
            limit_id: definition.limit_id.clone(),
            timestamp: timing
                .finished_at
                .clone()
                .unwrap_or(timing.recorded_at.clone()),
            spent: wall_seconds,
            reserved: 0.0,
            source_kind: "run_phase_timing".to_string(),
            source_id: timing.timing_id.clone(),
        });
    }
}

fn wall_seconds_from_timings_until(timings: &[RunPhaseTimingRecord], timestamp: &str) -> f64 {
    let cutoff = parse_ts_seconds(timestamp);
    timings
        .iter()
        .filter_map(|timing| {
            let observed_at = timing.finished_at.as_deref().unwrap_or(&timing.recorded_at);
            if let (Some(cutoff), Some(observed_ts)) = (cutoff, parse_ts_seconds(observed_at)) {
                if observed_ts > cutoff {
                    return None;
                }
            }
            timing
                .wall_seconds
                .filter(|value| value.is_finite() && *value > 0.0)
        })
        .sum()
}

fn progress_events(
    definitions: &[LimitDefinition],
    observations: &[LimitObservation],
) -> Vec<LimitProgressEvent> {
    let max_by_limit = definitions
        .iter()
        .map(|definition| (definition.limit_id.clone(), definition.max_value))
        .collect::<BTreeMap<_, _>>();
    let mut previous = BTreeMap::<String, (f64, f64)>::new();
    let mut events = Vec::new();
    for observation in observations {
        let Some(max_value) = max_by_limit.get(&observation.limit_id).copied() else {
            continue;
        };
        let (last_spent, last_reserved) = previous
            .get(&observation.limit_id)
            .copied()
            .unwrap_or((0.0, 0.0));
        let event = LimitProgressEvent {
            schema_version: LIMIT_ENGINE_SCHEMA_VERSION.to_string(),
            event_id: limit_event_id(observation),
            run_id: observation.run_id.clone(),
            limit_id: observation.limit_id.clone(),
            timestamp: observation.timestamp.clone(),
            spent: observation.spent,
            reserved: observation.reserved,
            remaining: remaining(max_value, observation.spent, observation.reserved),
            utilization: utilization(max_value, observation.spent, observation.reserved),
            delta_spent: (observation.spent - last_spent).max(0.0),
            delta_reserved: observation.reserved - last_reserved,
            source_kind: observation.source_kind.clone(),
            source_id: observation.source_id.clone(),
        };
        previous.insert(
            observation.limit_id.clone(),
            (observation.spent, observation.reserved),
        );
        events.push(event);
    }
    events
}

fn forecast_limit(
    definition: &LimitDefinition,
    spent: f64,
    remaining: f64,
    events: &[LimitProgressEvent],
    generated_at: &str,
) -> LimitForecast {
    if remaining <= 0.0 {
        return LimitForecast {
            schema_version: LIMIT_ENGINE_SCHEMA_VERSION.to_string(),
            forecast_id: limit_forecast_id(&definition.limit_id, generated_at),
            run_id: definition.run_id.clone(),
            limit_id: definition.limit_id.clone(),
            model: "exhausted".to_string(),
            predicted_crossing_at: Some(generated_at.to_string()),
            seconds_to_limit: Some(0.0),
            seconds_to_limit_low: Some(0.0),
            seconds_to_limit_high: Some(0.0),
            predicted_crossing_at_low: Some(generated_at.to_string()),
            predicted_crossing_at_high: Some(generated_at.to_string()),
            rate_per_second: None,
            confidence: ForecastConfidence::High,
            sample_count: events.len() as u64,
            updated_at: generated_at.to_string(),
        };
    }
    let samples = events
        .iter()
        .filter_map(|event| parse_ts_seconds(&event.timestamp).map(|ts| (ts, event.spent)))
        .filter(|(_, value)| value.is_finite() && *value >= 0.0)
        .collect::<Vec<_>>();
    let rates = positive_rates(&samples);
    let (model, rate, confidence) = forecast_rate(&rates);
    let rate = cap_rate_to_elapsed_average(rate, &samples);
    let (predicted_crossing_at, seconds_to_limit) = rate
        .filter(|value| *value > 0.0 && value.is_finite())
        .map(|rate| {
            let seconds = (remaining / rate).max(0.0);
            (add_seconds(generated_at, seconds), Some(seconds))
        })
        .unwrap_or((None, None));
    let seconds_to_limit_low = seconds_to_limit.map(|seconds| {
        let multiplier = interval_low_multiplier(confidence);
        (seconds * multiplier).max(0.0)
    });
    let seconds_to_limit_high = seconds_to_limit.map(|seconds| {
        let multiplier = interval_high_multiplier(confidence);
        (seconds * multiplier).max(0.0)
    });
    let predicted_crossing_at_low =
        seconds_to_limit_low.and_then(|seconds| add_seconds(generated_at, seconds));
    let predicted_crossing_at_high =
        seconds_to_limit_high.and_then(|seconds| add_seconds(generated_at, seconds));
    let rate_per_second = seconds_to_limit.map(|_| rate.unwrap_or(0.0));
    let confidence = if spent <= 0.0 && rates.is_empty() {
        ForecastConfidence::Unknown
    } else {
        confidence
    };
    LimitForecast {
        schema_version: LIMIT_ENGINE_SCHEMA_VERSION.to_string(),
        forecast_id: limit_forecast_id(&definition.limit_id, generated_at),
        run_id: definition.run_id.clone(),
        limit_id: definition.limit_id.clone(),
        model,
        predicted_crossing_at,
        seconds_to_limit,
        seconds_to_limit_low,
        seconds_to_limit_high,
        predicted_crossing_at_low,
        predicted_crossing_at_high,
        rate_per_second,
        confidence,
        sample_count: samples.len() as u64,
        updated_at: generated_at.to_string(),
    }
}

fn positive_rates(samples: &[(f64, f64)]) -> Vec<f64> {
    let mut rates = Vec::new();
    for pair in samples.windows(2) {
        let (previous_ts, previous_value) = pair[0];
        let (next_ts, next_value) = pair[1];
        let dt = next_ts - previous_ts;
        let delta = next_value - previous_value;
        if dt >= MIN_FORECAST_INTERVAL_SECONDS && delta > 0.0 {
            rates.push(delta / dt);
        }
    }
    rates
}

fn cap_rate_to_elapsed_average(rate: Option<f64>, samples: &[(f64, f64)]) -> Option<f64> {
    let rate = rate.filter(|value| *value > 0.0 && value.is_finite())?;
    let first = samples.first().copied()?;
    let last = samples.last().copied()?;
    let elapsed = last.0 - first.0;
    let delta = last.1 - first.1;
    if elapsed < MIN_FORECAST_INTERVAL_SECONDS || delta <= 0.0 {
        return Some(rate);
    }
    let elapsed_average = delta / elapsed;
    if elapsed_average <= 0.0 || !elapsed_average.is_finite() {
        return Some(rate);
    }
    Some(rate.min(elapsed_average * MAX_BURST_TO_ELAPSED_RATE_RATIO))
}

fn interval_low_multiplier(confidence: ForecastConfidence) -> f64 {
    match confidence {
        ForecastConfidence::High => 0.85,
        ForecastConfidence::Medium => FORECAST_INTERVAL_LOW_MULTIPLIER,
        ForecastConfidence::Low => 0.5,
        ForecastConfidence::Unknown => 0.0,
    }
}

fn interval_high_multiplier(confidence: ForecastConfidence) -> f64 {
    match confidence {
        ForecastConfidence::High => 1.2,
        ForecastConfidence::Medium => FORECAST_INTERVAL_HIGH_MULTIPLIER,
        ForecastConfidence::Low => 2.25,
        ForecastConfidence::Unknown => 0.0,
    }
}

fn forecast_rate(rates: &[f64]) -> (String, Option<f64>, ForecastConfidence) {
    if rates.is_empty() {
        return (
            "insufficient_samples".to_string(),
            None,
            ForecastConfidence::Unknown,
        );
    }
    if rates.len() < 3 {
        return (
            "ewma_fallback".to_string(),
            Some(ewma(rates, 0.45)),
            ForecastConfidence::Low,
        );
    }
    let mean = rates.iter().sum::<f64>() / rates.len() as f64;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for pair in rates.windows(2) {
        numerator += (pair[0] - mean) * (pair[1] - mean);
        denominator += (pair[0] - mean).powi(2);
    }
    let phi = if denominator > 0.0 {
        (numerator / denominator).clamp(-0.8, 0.95)
    } else {
        0.0
    };
    let last = rates.last().copied().unwrap_or(mean);
    let predicted = (mean + phi * (last - mean)).max(0.0);
    let rate = if predicted > 0.0 {
        predicted
    } else {
        ewma(rates, 0.35)
    };
    let confidence = if rates.len() >= 8 {
        ForecastConfidence::High
    } else {
        ForecastConfidence::Medium
    };
    ("ar1".to_string(), Some(rate), confidence)
}

fn ewma(values: &[f64], alpha: f64) -> f64 {
    let mut iter = values.iter();
    let Some(first) = iter.next().copied() else {
        return 0.0;
    };
    iter.fold(first, |acc, value| alpha * *value + (1.0 - alpha) * acc)
}

fn latest_observation_for<'a>(
    limit_id: &str,
    observations: &'a [LimitObservation],
) -> Option<&'a LimitObservation> {
    observations
        .iter()
        .rev()
        .find(|observation| observation.limit_id == limit_id)
}

fn ledger_values_for_kind(kind: &LimitKind, ledger: &BudgetLedgerSnapshot) -> (f64, f64) {
    match kind {
        LimitKind::CostUsd => (ledger.spent_cost_usd, ledger.reserved_cost_usd),
        LimitKind::PromptTokens => (
            ledger.spent_prompt_tokens as f64,
            ledger.reserved_prompt_tokens as f64,
        ),
        LimitKind::CompletionTokens => (
            ledger.spent_completion_tokens as f64,
            ledger.reserved_completion_tokens as f64,
        ),
        LimitKind::TotalTokens => (
            ledger.spent_total_tokens as f64,
            ledger.reserved_total_tokens as f64,
        ),
        LimitKind::TotalRollouts => (
            ledger.spent_rollouts as f64,
            ledger.reserved_rollouts as f64,
        ),
        LimitKind::WallSeconds => (
            ledger.spent_wall_seconds as f64,
            ledger.reserved_wall_seconds as f64,
        ),
        _ => (0.0, 0.0),
    }
}

fn reservation_value_for_kind(kind: &LimitKind, reservation: &BudgetReservationRecord) -> f64 {
    match kind {
        LimitKind::CostUsd => reservation.max_cost_usd.unwrap_or(0.0),
        LimitKind::PromptTokens => reservation.max_prompt_tokens.unwrap_or(0) as f64,
        LimitKind::CompletionTokens => reservation.max_completion_tokens.unwrap_or(0) as f64,
        // An absent total-token cap means this reservation reserves nothing against
        // the total-tokens limit; the caller skips a 0 value. Do NOT synthesize a
        // total from the prompt/completion caps — that fabricates a reservation the
        // caller never declared.
        LimitKind::TotalTokens => reservation.max_total_tokens.unwrap_or(0) as f64,
        LimitKind::TotalRollouts => reservation.max_rollouts.unwrap_or(0) as f64,
        LimitKind::WallSeconds => reservation.max_wall_seconds.unwrap_or(0) as f64,
        _ => 0.0,
    }
}

fn remaining(max_value: f64, spent: f64, reserved: f64) -> f64 {
    (max_value - spent - reserved).max(0.0)
}

fn utilization(max_value: f64, spent: f64, reserved: f64) -> f64 {
    if max_value <= 0.0 {
        return 0.0;
    }
    ((spent + reserved) / max_value).clamp(0.0, 1.0)
}

fn compare_observations(left: &LimitObservation, right: &LimitObservation) -> Ordering {
    left.timestamp
        .cmp(&right.timestamp)
        .then_with(|| left.limit_id.cmp(&right.limit_id))
        .then_with(|| left.source_kind.cmp(&right.source_kind))
        .then_with(|| left.source_id.cmp(&right.source_id))
}

fn parse_ts_seconds(value: &str) -> Option<f64> {
    OffsetDateTime::parse(value, &time::format_description::well_known::Rfc3339)
        .ok()
        .map(|ts| ts.unix_timestamp() as f64 + f64::from(ts.nanosecond()) / 1_000_000_000.0)
}

fn add_seconds(base: &str, seconds: f64) -> Option<String> {
    if !seconds.is_finite() {
        return None;
    }
    let base = OffsetDateTime::parse(base, &time::format_description::well_known::Rfc3339).ok()?;
    let seconds = seconds.ceil().min(i64::MAX as f64).max(0.0) as i64;
    (base + time::Duration::seconds(seconds))
        .format(&time::format_description::well_known::Rfc3339)
        .ok()
}

fn limit_event_id(observation: &LimitObservation) -> String {
    let identity = json!({
        "run_id": observation.run_id,
        "limit_id": observation.limit_id,
        "timestamp": observation.timestamp,
        "source_kind": observation.source_kind,
        "source_id": observation.source_id,
    });
    format!("limit_event_{}", &stable_value_hash(&identity)[..16])
}

fn limit_forecast_id(limit_id: &str, updated_at: &str) -> String {
    let identity = json!({
        "limit_id": limit_id,
        "updated_at": updated_at,
    });
    format!("limit_forecast_{}", &stable_value_hash(&identity)[..16])
}

fn now_rfc3339() -> String {
    OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}
