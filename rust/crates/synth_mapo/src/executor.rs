use serde_json::{json, Map, Value};
use synth_optimizer_platform::{ContainerClient, OptimizerError, Result};

use crate::candidate::{MapoBranchCheckpoint, MapoCandidate, MapoRolloutRecord};
use crate::config::MapoConfig;
use crate::runtime::MAPO_ALGORITHM_ID;

pub fn execute_candidate_rollouts(
    client: &ContainerClient,
    config: &MapoConfig,
    candidate: &MapoCandidate,
    split: &str,
    rollout_group: &str,
    seeds: &[i64],
    rollouts_per_candidate: usize,
) -> Result<Vec<MapoRolloutRecord>> {
    let mut records = Vec::new();
    for episode_index in 0..rollouts_per_candidate {
        for seed in seeds {
            let rollout_id = rollout_id(
                config,
                candidate,
                split,
                rollout_group,
                *seed,
                episode_index,
            );
            let request = rollout_request(
                config,
                candidate,
                split,
                rollout_group,
                *seed,
                episode_index,
                &rollout_id,
            );
            let response = client.rollout(&request)?;
            records.push(rollout_record(
                &rollout_id,
                candidate,
                split,
                rollout_group,
                *seed,
                episode_index,
                None,
                None,
                response,
            )?);
        }
    }
    Ok(records)
}

pub fn execute_candidate_task_rollouts(
    client: &ContainerClient,
    config: &MapoConfig,
    candidate: &MapoCandidate,
    split: &str,
    rollout_group: &str,
    task_instance_ids: &[String],
    rollouts_per_task: usize,
) -> Result<Vec<MapoRolloutRecord>> {
    let mut records = Vec::new();
    for episode_index in 0..rollouts_per_task {
        for task_instance_id in task_instance_ids {
            let seed = task_instance_seed(task_instance_id)?;
            let rollout_id =
                rollout_id(config, candidate, split, rollout_group, seed, episode_index);
            let mut request = rollout_request(
                config,
                candidate,
                split,
                rollout_group,
                seed,
                episode_index,
                &rollout_id,
            );
            apply_task_instance_id(&mut request, task_instance_id)?;
            let response = client.rollout(&request)?;
            records.push(rollout_record(
                &rollout_id,
                candidate,
                split,
                rollout_group,
                seed,
                episode_index,
                Some(task_instance_id.clone()),
                None,
                response,
            )?);
        }
    }
    Ok(records)
}

pub fn execute_branch_discovery_seed_rollouts(
    client: &ContainerClient,
    config: &MapoConfig,
    candidate: &MapoCandidate,
    split: &str,
    rollout_group: &str,
    seeds: &[i64],
) -> Result<(Vec<MapoRolloutRecord>, Vec<MapoBranchCheckpoint>)> {
    let mut records = Vec::new();
    let mut checkpoints = Vec::new();
    for seed in seeds {
        let rollout_id = rollout_id(config, candidate, split, rollout_group, *seed, 0);
        let mut request = rollout_request(
            config,
            candidate,
            split,
            rollout_group,
            *seed,
            0,
            &rollout_id,
        );
        set_request_steps(&mut request, config.mapo.branch_discovery_steps);
        attach_checkpoint_schedule(&mut request, &rollout_id);
        let response = client.rollout(&request)?;
        checkpoints.extend(extract_branch_checkpoints(
            &response,
            &rollout_id,
            *seed,
            None,
            config.mapo.branch_checkpoint_min_step,
            config.mapo.branch_checkpoints_per_rollout,
            &config.mapo.branch_checkpoint_strategy,
        )?);
        records.push(rollout_record(
            &rollout_id,
            candidate,
            split,
            rollout_group,
            *seed,
            0,
            None,
            None,
            response,
        )?);
    }
    Ok((records, checkpoints))
}

pub fn execute_branch_discovery_task_rollouts(
    client: &ContainerClient,
    config: &MapoConfig,
    candidate: &MapoCandidate,
    split: &str,
    rollout_group: &str,
    task_instance_ids: &[String],
) -> Result<(Vec<MapoRolloutRecord>, Vec<MapoBranchCheckpoint>)> {
    let mut records = Vec::new();
    let mut checkpoints = Vec::new();
    for task_instance_id in task_instance_ids {
        let seed = task_instance_seed(task_instance_id)?;
        let rollout_id = rollout_id(config, candidate, split, rollout_group, seed, 0);
        let mut request = rollout_request(
            config,
            candidate,
            split,
            rollout_group,
            seed,
            0,
            &rollout_id,
        );
        apply_task_instance_id(&mut request, task_instance_id)?;
        set_request_steps(&mut request, config.mapo.branch_discovery_steps);
        attach_checkpoint_schedule(&mut request, &rollout_id);
        let response = client.rollout(&request)?;
        checkpoints.extend(extract_branch_checkpoints(
            &response,
            &rollout_id,
            seed,
            Some(task_instance_id.clone()),
            config.mapo.branch_checkpoint_min_step,
            config.mapo.branch_checkpoints_per_rollout,
            &config.mapo.branch_checkpoint_strategy,
        )?);
        records.push(rollout_record(
            &rollout_id,
            candidate,
            split,
            rollout_group,
            seed,
            0,
            Some(task_instance_id.clone()),
            None,
            response,
        )?);
    }
    Ok((records, checkpoints))
}

pub fn execute_candidate_branch_rollouts(
    client: &ContainerClient,
    config: &MapoConfig,
    candidate: &MapoCandidate,
    split: &str,
    rollout_group: &str,
    checkpoints: &[MapoBranchCheckpoint],
) -> Result<Vec<MapoRolloutRecord>> {
    let mut records = Vec::new();
    for (episode_index, checkpoint) in checkpoints.iter().enumerate() {
        let rollout_id = format!(
            "{}_branch",
            rollout_id(
                config,
                candidate,
                split,
                rollout_group,
                checkpoint.seed,
                episode_index,
            )
        );
        let mut request = rollout_request(
            config,
            candidate,
            split,
            rollout_group,
            checkpoint.seed,
            episode_index,
            &rollout_id,
        );
        set_request_steps(&mut request, config.mapo.branch_rollout_steps);
        let response = client.resume_rollout(
            &checkpoint.parent_rollout_id,
            &checkpoint.checkpoint_id,
            &request,
        )?;
        records.push(rollout_record(
            &rollout_id,
            candidate,
            split,
            rollout_group,
            checkpoint.seed,
            episode_index,
            checkpoint.task_instance_id.clone(),
            Some(checkpoint),
            response,
        )?);
    }
    Ok(records)
}

pub fn rollout_request(
    config: &MapoConfig,
    candidate: &MapoCandidate,
    split: &str,
    rollout_group: &str,
    seed: i64,
    episode_index: usize,
    rollout_id: &str,
) -> Value {
    let mut env_config = config.taskset.env_config.clone();
    env_config.insert(
        "communication_protocol".to_string(),
        candidate.protocol_value(),
    );
    env_config.insert("max_steps".to_string(), json!(config.mapo.max_steps));
    env_config.insert("segment_steps".to_string(), json!(config.mapo.max_steps));
    env_config.insert("seed".to_string(), json!(seed));

    let mut policy_config = Map::new();
    policy_config.insert("kind".to_string(), json!("openai"));
    policy_config.insert("provider".to_string(), json!(&config.policy.provider));
    policy_config.insert("model".to_string(), json!(&config.policy.model));
    if let Some(api_key_env) = config.policy.api_key_env.as_deref() {
        policy_config.insert("api_key_env".to_string(), json!(api_key_env));
    }
    if let Some(inference_url) = config.policy.inference_url.as_deref() {
        policy_config.insert("inference_url".to_string(), json!(inference_url));
    }
    if let Some(max_tokens) = config.policy.max_tokens {
        policy_config.insert("max_tokens".to_string(), json!(max_tokens));
    }
    if !candidate.roles.is_empty() {
        policy_config.insert("role_prompts".to_string(), json!(&candidate.roles));
    }
    for (key, value) in &config.policy.config {
        policy_config.insert(key.clone(), value.clone());
    }

    let mut env = Map::new();
    env.insert("seed".to_string(), json!(seed));
    env.insert("config".to_string(), Value::Object(env_config));

    let mut request = json!({
        "rollout_id": rollout_id,
        "trace_correlation_id": rollout_id,
        "submission_mode": "sync",
        "env": Value::Object(env),
        "policy": {
            "provider": &config.policy.provider,
            "model": &config.policy.model,
            "config": Value::Object(policy_config),
        },
        "metadata": {
            "algorithm_id": MAPO_ALGORITHM_ID,
            "run_id": &config.run.run_id,
            "candidate_id": &candidate.id,
            "generation": candidate.generation,
            "split": split,
            "rollout_group": rollout_group,
            "seed": seed,
            "episode_index": episode_index,
        },
    });
    if let Some(task_instance_id) = task_instance_id(config, split, seed) {
        request["task_instance_id"] = json!(task_instance_id);
    }
    request
}

fn set_request_steps(request: &mut Value, steps: usize) {
    request["max_steps"] = json!(steps);
    request["segment_steps"] = json!(steps);
    request["env"]["config"]["max_steps"] = json!(steps);
    request["env"]["config"]["segment_steps"] = json!(steps);
}

fn attach_checkpoint_schedule(request: &mut Value, rollout_id: &str) {
    request["checkpoint_schedule"] = json!({
        "mode": "per_llm_call",
        "checkpoint_id_prefix": format!("{rollout_id}_branch_cp"),
    });
}

fn apply_task_instance_id(request: &mut Value, task_instance_id: &str) -> Result<()> {
    request["task_instance_id"] = json!(task_instance_id);
    request["metadata"]["task_instance_id"] = json!(task_instance_id);
    if let Some(quest_id) = dungeongrid_plus_quest_id(task_instance_id)? {
        request["env"]["config"]["quest_id"] = json!(quest_id);
    }
    Ok(())
}

fn dungeongrid_plus_quest_id(task_instance_id: &str) -> Result<Option<String>> {
    if !task_instance_id.starts_with("dungeongrid_plus:") {
        return Ok(None);
    }
    let parts = task_instance_id.split(':').collect::<Vec<_>>();
    if parts.len() < 5 {
        return Err(OptimizerError::Config(format!(
            "DungeongridPlus task instance id must include split, quest id, and seed: {task_instance_id}"
        )));
    }
    Ok(Some(parts[2..parts.len() - 1].join(":")))
}

fn task_instance_id(config: &MapoConfig, split: &str, seed: i64) -> Option<String> {
    let task_split = task_instance_split(split);
    if let Some(template) = config.taskset.task_instance_template.as_deref() {
        return Some(
            template
                .replace("{split}", task_split)
                .replace("{seed}", &seed.to_string()),
        );
    }
    config.taskset.task_instance_id.clone()
}

fn task_instance_split(split: &str) -> &str {
    if split == "heldout_baseline" {
        "heldout"
    } else if split == "selection" {
        "train"
    } else {
        split
    }
}

fn task_instance_seed(task_instance_id: &str) -> Result<i64> {
    task_instance_id
        .rsplit(':')
        .next()
        .and_then(|value| value.parse::<i64>().ok())
        .ok_or_else(|| {
            OptimizerError::Config(format!(
                "MAPO selection task instance id must end in numeric seed: {task_instance_id}"
            ))
        })
}

fn rollout_id(
    config: &MapoConfig,
    candidate: &MapoCandidate,
    split: &str,
    rollout_group: &str,
    seed: i64,
    episode_index: usize,
) -> String {
    format!(
        "{}_{}_{}_{}_{}_{}",
        sanitize_id(&config.run.run_id),
        sanitize_id(&candidate.id),
        sanitize_id(split),
        sanitize_id(rollout_group),
        seed,
        episode_index
    )
}

fn extract_branch_checkpoints(
    response: &Value,
    parent_rollout_id: &str,
    seed: i64,
    task_instance_id: Option<String>,
    min_step: usize,
    keep: usize,
    strategy: &str,
) -> Result<Vec<MapoBranchCheckpoint>> {
    let scheduled = response
        .pointer("/metadata/scheduled_checkpoints")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            OptimizerError::Container(format!(
                "MAPO discovery rollout {parent_rollout_id} did not return scheduled checkpoints"
            ))
        })?;
    let mut checkpoints = scheduled
        .iter()
        .filter_map(|item| {
            let checkpoint_id = item.get("checkpoint_id")?.as_str()?.trim();
            if checkpoint_id.is_empty() {
                return None;
            }
            let step = item
                .get("step")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .unwrap_or(0);
            if step < min_step {
                return None;
            }
            let reward = item.get("reward").and_then(Value::as_f64).unwrap_or(0.0);
            let message_metrics = item
                .get("message_metrics")
                .and_then(Value::as_object)
                .unwrap_or(item.as_object()?);
            Some(MapoBranchCheckpoint {
                checkpoint_id: checkpoint_id.to_string(),
                parent_rollout_id: parent_rollout_id.to_string(),
                seed,
                task_instance_id: task_instance_id.clone(),
                step,
                reward,
                messages_delivered: metric_u64(message_metrics, "messages_delivered"),
                messages_rejected: metric_u64(message_metrics, "messages_rejected"),
                message_chars: metric_u64(message_metrics, "message_chars"),
            })
        })
        .collect::<Vec<_>>();
    checkpoints.sort_by(|left, right| checkpoint_order(left, right, strategy));
    checkpoints.truncate(keep);
    if checkpoints.is_empty() {
        return Err(OptimizerError::Container(format!(
            "MAPO discovery rollout {parent_rollout_id} produced no branch checkpoints at or after step {min_step}"
        )));
    }
    Ok(checkpoints)
}

fn checkpoint_order(
    left: &MapoBranchCheckpoint,
    right: &MapoBranchCheckpoint,
    strategy: &str,
) -> std::cmp::Ordering {
    let reward_desc = || {
        right
            .reward
            .partial_cmp(&left.reward)
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    match strategy {
        "early" => left
            .step
            .cmp(&right.step)
            .then_with(reward_desc)
            .then_with(|| left.checkpoint_id.cmp(&right.checkpoint_id)),
        "late" => right
            .step
            .cmp(&left.step)
            .then_with(reward_desc)
            .then_with(|| left.checkpoint_id.cmp(&right.checkpoint_id)),
        _ => reward_desc()
            .then_with(|| right.step.cmp(&left.step))
            .then_with(|| left.checkpoint_id.cmp(&right.checkpoint_id)),
    }
}

fn rollout_record(
    rollout_id: &str,
    candidate: &MapoCandidate,
    split: &str,
    rollout_group: &str,
    seed: i64,
    episode_index: usize,
    task_instance_id: Option<String>,
    branch_parent: Option<&MapoBranchCheckpoint>,
    response: Value,
) -> Result<MapoRolloutRecord> {
    let summary = response
        .get("summary")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            OptimizerError::Container(format!(
                "MAPO rollout {rollout_id} response missing summary"
            ))
        })?;
    let communication = summary
        .get("message_metrics")
        .or_else(|| summary.get("communication"))
        .and_then(Value::as_object)
        .unwrap_or(summary);
    let raw_reward = summary
        .get("optimizer_score")
        .or_else(|| summary.get("outcome_reward"))
        .or_else(|| summary.get("total_reward"))
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let raw_messages_delivered = metric_u64(communication, "messages_delivered");
    let raw_messages_rejected = metric_u64(communication, "messages_rejected");
    let raw_message_chars = metric_u64(communication, "message_chars");
    let reward = branch_parent
        .map(|checkpoint| raw_reward - checkpoint.reward)
        .unwrap_or(raw_reward);
    let messages_delivered = branch_parent
        .map(|checkpoint| raw_messages_delivered.saturating_sub(checkpoint.messages_delivered))
        .unwrap_or(raw_messages_delivered);
    let messages_rejected = branch_parent
        .map(|checkpoint| raw_messages_rejected.saturating_sub(checkpoint.messages_rejected))
        .unwrap_or(raw_messages_rejected);
    let message_chars = branch_parent
        .map(|checkpoint| raw_message_chars.saturating_sub(checkpoint.message_chars))
        .unwrap_or(raw_message_chars);
    Ok(MapoRolloutRecord {
        rollout_id: rollout_id.to_string(),
        candidate_id: candidate.id.clone(),
        split: split.to_string(),
        rollout_group: rollout_group.to_string(),
        seed,
        episode_index,
        task_instance_id,
        parent_rollout_id: response
            .get("parent_rollout_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        parent_checkpoint_id: response
            .get("parent_checkpoint_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        checkpoint_id: response
            .get("checkpoint")
            .and_then(|checkpoint| checkpoint.get("checkpoint_id"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| {
                summary
                    .get("checkpoint_id")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            }),
        success: summary
            .get("success")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        reward,
        messages_delivered,
        messages_rejected,
        message_chars,
        response: compact_rollout_response(&response),
    })
}

fn metric_u64(summary: &Map<String, Value>, key: &str) -> u64 {
    summary.get(key).and_then(Value::as_u64).unwrap_or(0)
}

fn compact_rollout_response(response: &Value) -> Value {
    let usage = response.get("usage").or_else(|| {
        response
            .get("summary")
            .and_then(|summary| summary.get("usage"))
    });
    public_evidence_projection(&json!({
        "rollout_id": response.get("rollout_id"),
        "trace_correlation_id": response.get("trace_correlation_id"),
        "status": response.get("status"),
        "success_status": response.get("success_status"),
        "status_detail": response.get("status_detail"),
        "summary": response.get("summary"),
        "reward_info": response.get("reward_info"),
        "usage": usage,
        "metadata": response.get("metadata"),
        "parent_rollout_id": response.get("parent_rollout_id"),
        "parent_checkpoint_id": response.get("parent_checkpoint_id"),
        "checkpoint_id": response
            .get("checkpoint")
            .and_then(|checkpoint| checkpoint.get("checkpoint_id")),
    }))
}

fn public_evidence_projection(value: &Value) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .filter(|(key, _)| !is_private_evidence_key(key))
                .map(|(key, value)| (key.clone(), public_evidence_projection(value)))
                .collect(),
        ),
        Value::Array(values) => {
            Value::Array(values.iter().map(public_evidence_projection).collect())
        }
        _ => value.clone(),
    }
}

fn is_private_evidence_key(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "reason"
            | "rationale"
            | "private_rationale"
            | "private_reasoning"
            | "thought"
            | "thoughts"
            | "analysis"
            | "chain_of_thought"
            | "scratchpad"
    )
}

fn sanitize_id(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}
