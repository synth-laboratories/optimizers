use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::time::{SystemTime, UNIX_EPOCH};
use synth_optimizer_platform::GepaConfig;

use crate::planner::GepaCursor;

pub const EPISODE_METADATA_KEY: &str = "episode";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EpisodeOrigin {
    pub generation: usize,
    pub rollout_count: usize,
    pub cost_usd: f64,
    pub started_unix_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EpisodeCursorState {
    pub origin: EpisodeOrigin,
    #[serde(default)]
    pub proposer_rounds: usize,
    #[serde(default)]
    pub stop_reason: Option<String>,
}

impl EpisodeCursorState {
    pub fn from_cursor_baseline(cursor: &GepaCursor) -> Self {
        Self {
            origin: EpisodeOrigin {
                generation: cursor.generation,
                rollout_count: cursor.rollout_count,
                cost_usd: cursor.cost_usd,
                started_unix_ms: now_unix_ms(),
            },
            proposer_rounds: 0,
            stop_reason: None,
        }
    }
}

pub fn read_episode(cursor: &GepaCursor) -> Option<EpisodeCursorState> {
    let value = cursor.metadata.get(EPISODE_METADATA_KEY)?;
    serde_json::from_value(value.clone()).ok()
}

pub fn write_episode(cursor: &mut GepaCursor, episode: &EpisodeCursorState) {
    let mut metadata = match &cursor.metadata {
        Value::Object(map) => map.clone(),
        _ => Map::new(),
    };
    metadata.insert(
        EPISODE_METADATA_KEY.to_string(),
        serde_json::to_value(episode).unwrap_or_else(|_| json!({})),
    );
    cursor.metadata = Value::Object(metadata);
}

pub fn copy_episode_into(metadata: &mut Map<String, Value>, cursor: &GepaCursor) {
    if let Some(episode) = cursor.metadata.get(EPISODE_METADATA_KEY).cloned() {
        metadata.insert(EPISODE_METADATA_KEY.to_string(), episode);
    }
}

/// Stamp a new origin. Used on fixture fork so deltas start at this restart.
pub fn reset_episode_origin(cursor: &mut GepaCursor) {
    write_episode(cursor, &EpisodeCursorState::from_cursor_baseline(cursor));
}

/// Keep an existing origin across persist/restore of the same episode.
pub fn pin_episode_origin_if_missing(cursor: &mut GepaCursor) {
    if read_episode(cursor).is_some() {
        return;
    }
    reset_episode_origin(cursor);
}

pub fn increment_proposer_rounds(cursor: &mut GepaCursor) {
    pin_episode_origin_if_missing(cursor);
    let mut episode =
        read_episode(cursor).unwrap_or_else(|| EpisodeCursorState::from_cursor_baseline(cursor));
    episode.proposer_rounds = episode.proposer_rounds.saturating_add(1);
    write_episode(cursor, &episode);
}

pub fn record_stop_reason(cursor: &mut GepaCursor, reason: &str) {
    pin_episode_origin_if_missing(cursor);
    let mut episode =
        read_episode(cursor).unwrap_or_else(|| EpisodeCursorState::from_cursor_baseline(cursor));
    episode.stop_reason = Some(reason.to_string());
    write_episode(cursor, &episode);
}

/// First matching horizon. Episode limits are delta-from-restart and replace
/// the matching absolute budget when set.
pub fn horizon_reason(
    gepa: &GepaConfig,
    cursor: &GepaCursor,
    rollout_count: usize,
    cost_usd: f64,
) -> Option<&'static str> {
    let episode =
        read_episode(cursor).unwrap_or_else(|| EpisodeCursorState::from_cursor_baseline(cursor));
    if gepa.episode.has_delta_limits() {
        if let Some(limit) = gepa.episode.proposer_rounds {
            if episode.proposer_rounds >= limit {
                return Some("episode_proposer_rounds");
            }
        }
        if let Some(limit) = gepa.episode.max_rollouts {
            if rollout_count.saturating_sub(episode.origin.rollout_count) >= limit {
                return Some("episode_max_rollouts");
            }
        }
        if let Some(limit) = gepa.episode.max_spend_usd {
            if (cost_usd - episode.origin.cost_usd) >= limit {
                return Some("episode_max_spend_usd");
            }
        }
        if let Some(limit) = gepa.episode.max_wall_seconds {
            let elapsed_seconds =
                now_unix_ms().saturating_sub(episode.origin.started_unix_ms) / 1000;
            if elapsed_seconds >= limit {
                return Some("episode_max_wall_seconds");
            }
        }
        return None;
    }

    if cursor.generation >= gepa.max_generations {
        return Some("max_generations");
    }
    if rollout_count >= gepa.train_rollout_limit() {
        return Some("train_rollout_budget");
    }
    if gepa.max_cost_usd > 0.0 && cost_usd >= gepa.max_cost_usd {
        return Some("cost_budget");
    }
    None
}

pub fn horizon_kind(reason: &str) -> Value {
    match reason {
        "episode_proposer_rounds" => json!({"kind": "episode_proposer_rounds"}),
        "episode_max_rollouts" => json!({"kind": "episode_max_rollouts"}),
        "episode_max_spend_usd" => json!({"kind": "episode_max_spend_usd"}),
        "episode_max_wall_seconds" => json!({"kind": "episode_max_wall_seconds"}),
        "max_generations" => json!({"kind": "max_generations"}),
        "train_rollout_budget" => json!({"kind": "max_rollouts"}),
        "cost_budget" => json!({"kind": "max_cost_usd"}),
        other => json!({"kind": other}),
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use synth_optimizer_platform::GepaEpisodeConfig;

    fn config_with_episode(episode: GepaEpisodeConfig) -> GepaConfig {
        let mut gepa = GepaConfig::default();
        gepa.episode = episode;
        gepa.max_generations = 2;
        gepa.max_total_rollouts = 16;
        gepa
    }

    #[test]
    fn three_proposer_rounds_after_a_gen1_fork_ignores_absolute_generation_ceiling() {
        let mut cursor = GepaCursor::new("child");
        cursor.generation = 1;
        cursor.rollout_count = 1040;
        reset_episode_origin(&mut cursor);
        increment_proposer_rounds(&mut cursor);
        increment_proposer_rounds(&mut cursor);
        let gepa = config_with_episode(GepaEpisodeConfig {
            proposer_rounds: Some(3),
            skip_heldout: true,
            ..GepaEpisodeConfig::default()
        });
        assert_eq!(
            horizon_reason(&gepa, &cursor, cursor.rollout_count, 0.0),
            None,
            "two completed rounds of a three-round episode must still admit"
        );
        increment_proposer_rounds(&mut cursor);
        assert_eq!(
            horizon_reason(&gepa, &cursor, cursor.rollout_count, 0.0),
            Some("episode_proposer_rounds")
        );
        assert_eq!(cursor.generation, 1);
        assert_eq!(gepa.max_generations, 2);
    }

    #[test]
    fn rollout_and_spend_limits_are_delta_from_fork() {
        let mut cursor = GepaCursor::new("child");
        cursor.rollout_count = 1000;
        cursor.cost_usd = 4.0;
        reset_episode_origin(&mut cursor);
        let gepa = config_with_episode(GepaEpisodeConfig {
            max_rollouts: Some(50),
            max_spend_usd: Some(1.0),
            ..GepaEpisodeConfig::default()
        });
        assert_eq!(horizon_reason(&gepa, &cursor, 1040, 4.4), None);
        assert_eq!(
            horizon_reason(&gepa, &cursor, 1050, 4.4),
            Some("episode_max_rollouts")
        );
        let spend_only = config_with_episode(GepaEpisodeConfig {
            max_spend_usd: Some(1.0),
            ..GepaEpisodeConfig::default()
        });
        assert_eq!(
            horizon_reason(&spend_only, &cursor, 1040, 5.0),
            Some("episode_max_spend_usd")
        );
    }

    #[test]
    fn unset_episode_still_uses_absolute_generation_ceiling() {
        let mut cursor = GepaCursor::new("fresh");
        cursor.generation = 2;
        pin_episode_origin_if_missing(&mut cursor);
        let gepa = config_with_episode(GepaEpisodeConfig::default());
        assert_eq!(
            horizon_reason(&gepa, &cursor, 0, 0.0),
            Some("max_generations")
        );
    }
}
