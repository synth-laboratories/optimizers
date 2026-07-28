use serde::{Deserialize, Serialize};

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
    ) -> Self {
        let baseline_score = MapoScore::from_rollouts(baseline_records);
        let champion_score = MapoScore::from_rollouts(champion_records);
        let paired_episodes_per_arm = baseline_score
            .rollout_count
            .min(champion_score.rollout_count);
        let success_delta_pp = (champion_score.success_rate - baseline_score.success_rate) * 100.0;
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
        Self {
            baseline_candidate_id: baseline_candidate_id.into(),
            champion_candidate_id: champion_candidate_id.into(),
            baseline_score,
            champion_score,
            paired_episodes_per_arm,
            min_paired_episodes_per_arm,
            success_delta_pp,
            min_success_delta_pp,
            meets_episode_count_bar,
            meets_success_delta_bar,
            chars_per_success_not_worse,
            h2_bar_passed,
        }
    }
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
