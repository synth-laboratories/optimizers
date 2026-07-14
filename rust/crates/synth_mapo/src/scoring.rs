use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use synth_optimizer_platform::{OptimizerError, Result};

use crate::candidate::MapoRolloutRecord;

const EPSILON: f64 = 1e-9;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MapoScore {
    pub rollout_count: usize,
    pub success_count: usize,
    pub success_rate: f64,
    pub reward_mean: f64,
    pub messages_delivered: u64,
    pub messages_rejected: u64,
    pub message_chars: u64,
    pub chars_per_success: Option<f64>,
}

impl MapoScore {
    pub fn from_rollouts(records: &[MapoRolloutRecord]) -> Self {
        let rollout_count = records.len();
        let success_count = records.iter().filter(|record| record.success).count();
        let reward_total = records.iter().map(|record| record.reward).sum::<f64>();
        let messages_delivered = records
            .iter()
            .map(|record| record.messages_delivered)
            .sum::<u64>();
        let messages_rejected = records
            .iter()
            .map(|record| record.messages_rejected)
            .sum::<u64>();
        let message_chars = records
            .iter()
            .map(|record| record.message_chars)
            .sum::<u64>();
        let success_rate = if rollout_count == 0 {
            0.0
        } else {
            success_count as f64 / rollout_count as f64
        };
        let chars_per_success = if success_count == 0 {
            None
        } else {
            Some(message_chars as f64 / success_count as f64)
        };
        Self {
            rollout_count,
            success_count,
            success_rate,
            reward_mean: if rollout_count == 0 {
                0.0
            } else {
                reward_total / rollout_count as f64
            },
            messages_delivered,
            messages_rejected,
            message_chars,
            chars_per_success,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MapoHeldoutComparison {
    pub baseline_candidate_id: String,
    pub champion_candidate_id: String,
    pub baseline_score: MapoScore,
    pub champion_score: MapoScore,
    pub paired_episodes_per_arm: usize,
    pub pair_key_fields: [String; 3],
    pub pairing_status: String,
    pub success_delta_ci95_pp: Option<[f64; 2]>,
    pub reward_delta: f64,
    pub reward_delta_ci95: Option<[f64; 2]>,
    pub min_paired_episodes_per_arm: usize,
    pub success_delta_pp: f64,
    pub min_success_delta_pp: f64,
    pub meets_episode_count_bar: bool,
    pub meets_success_delta_bar: bool,
    pub chars_per_success_not_worse: Option<bool>,
    pub h2_bar_passed: bool,
}

impl MapoHeldoutComparison {
    pub fn new(
        baseline_candidate_id: impl Into<String>,
        champion_candidate_id: impl Into<String>,
        baseline_records: &[MapoRolloutRecord],
        champion_records: &[MapoRolloutRecord],
        min_paired_episodes_per_arm: usize,
        min_success_delta_pp: f64,
    ) -> Result<Self> {
        let baseline_by_key = records_by_pair_key("baseline", baseline_records)?;
        let champion_by_key = records_by_pair_key("champion", champion_records)?;
        if baseline_by_key.keys().ne(champion_by_key.keys()) {
            let baseline_only = baseline_by_key
                .keys()
                .filter(|key| !champion_by_key.contains_key(*key))
                .take(5)
                .map(MapoPairKey::display)
                .collect::<Vec<_>>();
            let champion_only = champion_by_key
                .keys()
                .filter(|key| !baseline_by_key.contains_key(*key))
                .take(5)
                .map(MapoPairKey::display)
                .collect::<Vec<_>>();
            return Err(OptimizerError::Invariant(format!(
                "MAPO heldout arms require exact pair keys; baseline_only={baseline_only:?} champion_only={champion_only:?}"
            )));
        }
        let baseline_score = MapoScore::from_rollouts(baseline_records);
        let champion_score = MapoScore::from_rollouts(champion_records);
        let paired_episodes_per_arm = baseline_by_key.len();
        let success_deltas_pp = baseline_by_key
            .iter()
            .map(|(key, baseline)| {
                let champion = champion_by_key[key];
                let champion_success = if champion.success { 1.0 } else { 0.0 };
                let baseline_success = if baseline.success { 1.0 } else { 0.0 };
                (champion_success - baseline_success) * 100.0
            })
            .collect::<Vec<_>>();
        let reward_deltas = baseline_by_key
            .iter()
            .map(|(key, baseline)| champion_by_key[key].reward - baseline.reward)
            .collect::<Vec<_>>();
        let (success_delta_pp, success_delta_ci95_pp) = mean_ci95(&success_deltas_pp);
        let (reward_delta, reward_delta_ci95) = mean_ci95(&reward_deltas);
        let meets_episode_count_bar = paired_episodes_per_arm >= min_paired_episodes_per_arm;
        let meets_success_delta_bar = success_delta_pp + EPSILON >= min_success_delta_pp;
        let chars_per_success_not_worse = match (
            champion_score.chars_per_success,
            baseline_score.chars_per_success,
        ) {
            (Some(champion), Some(baseline)) => Some(champion <= baseline + EPSILON),
            (Some(_), None) => Some(true),
            (None, Some(_)) => Some(true),
            _ => None,
        };
        let h2_bar_passed = meets_episode_count_bar
            && meets_success_delta_bar
            && chars_per_success_not_worse.unwrap_or(false);
        Ok(Self {
            baseline_candidate_id: baseline_candidate_id.into(),
            champion_candidate_id: champion_candidate_id.into(),
            baseline_score,
            champion_score,
            paired_episodes_per_arm,
            pair_key_fields: [
                "seed".to_string(),
                "episode_index".to_string(),
                "task_instance_id".to_string(),
            ],
            pairing_status: "exact".to_string(),
            success_delta_ci95_pp,
            reward_delta,
            reward_delta_ci95,
            min_paired_episodes_per_arm,
            success_delta_pp,
            min_success_delta_pp,
            meets_episode_count_bar,
            meets_success_delta_bar,
            chars_per_success_not_worse,
            h2_bar_passed,
        })
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct MapoPairKey {
    seed: i64,
    episode_index: usize,
    task_instance_id: String,
}

impl MapoPairKey {
    fn from_record(record: &MapoRolloutRecord) -> Self {
        Self {
            seed: record.seed,
            episode_index: record.episode_index,
            task_instance_id: record
                .task_instance_id
                .clone()
                .unwrap_or_else(|| format!("seed:{}", record.seed)),
        }
    }

    fn display(&self) -> String {
        format!(
            "({}, {}, {})",
            self.seed, self.episode_index, self.task_instance_id
        )
    }
}

fn records_by_pair_key<'a>(
    arm: &str,
    records: &'a [MapoRolloutRecord],
) -> Result<BTreeMap<MapoPairKey, &'a MapoRolloutRecord>> {
    if records.is_empty() {
        return Err(OptimizerError::Invariant(format!(
            "MAPO {arm} heldout arm has no records"
        )));
    }
    let mut indexed = BTreeMap::new();
    for record in records {
        let key = MapoPairKey::from_record(record);
        if indexed.insert(key.clone(), record).is_some() {
            return Err(OptimizerError::Invariant(format!(
                "duplicate MAPO {arm} heldout pair key {}",
                key.display()
            )));
        }
    }
    Ok(indexed)
}

fn mean_ci95(values: &[f64]) -> (f64, Option<[f64; 2]>) {
    if values.is_empty() {
        return (0.0, None);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if values.len() < 2 {
        return (mean, None);
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let half_width = 1.96 * (variance / values.len() as f64).sqrt();
    (mean, Some([mean - half_width, mean + half_width]))
}

pub fn mapo_score_better(left: &MapoScore, right: &MapoScore) -> bool {
    if (left.success_rate - right.success_rate).abs() > EPSILON {
        return left.success_rate > right.success_rate;
    }
    if (left.reward_mean - right.reward_mean).abs() > EPSILON {
        return left.reward_mean > right.reward_mean;
    }
    if left.message_chars != right.message_chars {
        return left.message_chars < right.message_chars;
    }
    false
}
