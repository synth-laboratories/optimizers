//! Optional operator surfaces in the proposer workspace.
//!
//! Default-off. When a nested `[gepa.operator.*]` block is enabled, write the
//! corresponding files under `state/` so the Codex proposer can read them on
//! the same path it already uses for jesterky / metadata. Manderqueue talks
//! the `mq-sdk` HTTP contract (blocking reqwest; the SDK itself is async).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{json, Value};
use synth_optimizer_platform::{
    GepaOperatorConfig, ManderqueueConfig, OptimizerError, Result, SynthOptimizerConfig,
};

pub fn prepare_operator_workspace(
    config: &SynthOptimizerConfig,
    workspace_dir: &Path,
) -> Result<()> {
    let operator = &config.gepa.operator;
    if !operator.any_enabled()
        && !config.proposer.mcp.enabled
        && config.proposer.prompt.style_guides.is_empty()
    {
        return Ok(());
    }
    let state_dir = workspace_dir.join("state");
    fs::create_dir_all(&state_dir).map_err(|source| OptimizerError::io(&state_dir, source))?;
    write_json(
        &state_dir.join("operator.json"),
        &json!({
            "schema_version": "gepa_operator.v1",
            "manderqueue": operator.manderqueue,
            "scratchpad": operator.scratchpad,
            "hypotheses": operator.hypotheses,
            "control": operator.control,
            "levers": operator.levers,
            "reward": operator.reward,
            "mcp_agent": {
                "enabled": operator.mcp_agent.enabled || config.proposer.mcp.enabled,
                "command": operator.mcp_agent.command.as_ref().or(config.proposer.mcp.command.as_ref()),
                "server": operator.mcp_agent.server.as_ref().or(config.proposer.mcp.server.as_ref()),
            },
            "schema_repair_rounds": config.proposer.schema_repair_rounds,
            "pause": "/runs/{id}/pause",
            "resume": "/runs/{id}/resume",
            "branch": "POST /runs with fork_from",
        }),
    )?;
    if operator.scratchpad.enabled {
        write_scratchpad(&state_dir, operator)?;
    }
    if operator.hypotheses.enabled {
        write_hypotheses(&state_dir, operator)?;
    }
    if operator.manderqueue.enabled {
        sync_manderqueue_inbox(&state_dir, &operator.manderqueue, &config.run.run_id)?;
    }
    if operator.mcp_agent.enabled || config.proposer.mcp.enabled {
        let mcp = if config.proposer.mcp.enabled {
            &config.proposer.mcp
        } else {
            &operator.mcp_agent
        };
        write_json(
            &state_dir.join("mcp_agent.json"),
            &json!({
                "schema_version": "gepa_mcp_agent.v1",
                "enabled": true,
                "command": mcp.command,
                "server": mcp.server,
                "use": "Optional external agent. Call only the named MCP server; do not invent tools."
            }),
        )?;
    }
    Ok(())
}

pub fn operator_workspace_rule(operator: &GepaOperatorConfig) -> String {
    let mut lines = Vec::new();
    if operator.scratchpad.enabled {
        lines.push(format!(
            "Read and append to `{}` as the shared run scratchpad. Do not rewrite other agents' notes.",
            operator.scratchpad.path
        ));
    }
    if operator.hypotheses.enabled {
        lines.push(
            "Update `state/hypotheses.json` when you form, confirm, or retire a candidate hypothesis."
                .to_string(),
        );
    }
    if operator.manderqueue.enabled {
        lines.push(
            "Read `state/guidance.md` and `state/manderqueue_inbox.json` for operator messages. Reply via the same thread if you need a human."
                .to_string(),
        );
    }
    if operator.mcp_agent.enabled {
        lines.push(
            "If `state/mcp_agent.json` is present, that MCP server is an allowed external agent."
                .to_string(),
        );
    }
    lines.join("\n")
}

pub fn harvest_operator_workspace(workspace_dir: &Path) -> Result<Value> {
    let state_dir = workspace_dir.join("state");
    let hypotheses = read_json_if_exists(&state_dir.join("hypotheses.json"))?;
    let inbox = read_json_if_exists(&state_dir.join("manderqueue_inbox.json"))?;
    let operator = read_json_if_exists(&state_dir.join("operator.json"))?;
    let mcp = read_json_if_exists(&state_dir.join("mcp_agent.json"))?;
    let scratchpad = state_dir.join("scratchpad.md");
    Ok(json!({
        "operator": operator,
        "hypotheses": hypotheses,
        "manderqueue_inbox": inbox,
        "mcp_agent": mcp,
        "scratchpad_exists": scratchpad.exists(),
        "guidance_exists": state_dir.join("guidance.md").exists(),
    }))
}

fn read_json_if_exists(path: &Path) -> Result<Value> {
    if !path.exists() {
        return Ok(Value::Null);
    }
    let text = fs::read_to_string(path).map_err(|source| OptimizerError::io(path, source))?;
    Ok(serde_json::from_str(&text).unwrap_or(Value::Null))
}

fn write_scratchpad(state_dir: &Path, operator: &GepaOperatorConfig) -> Result<()> {
    let rel = operator.scratchpad.path.trim();
    let path = if rel.starts_with("state/") {
        state_dir.join(rel.trim_start_matches("state/"))
    } else {
        state_dir.join(rel)
    };
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| OptimizerError::io(parent, source))?;
    }
    let body = "# GEPA shared scratchpad\n\nAppend dated notes. Do not delete other entries.\n";
    fs::write(&path, body).map_err(|source| OptimizerError::io(&path, source))
}

fn write_hypotheses(state_dir: &Path, operator: &GepaOperatorConfig) -> Result<()> {
    let path = state_dir.join("hypotheses.json");
    if path.exists() {
        return Ok(());
    }
    write_json(
        &path,
        &json!({
            "schema_version": "gepa_hypotheses.v1",
            "max_open": operator.hypotheses.max_open,
            "open": [],
            "retired": [],
        }),
    )
}

fn sync_manderqueue_inbox(
    state_dir: &Path,
    mq: &ManderqueueConfig,
    run_id: &str,
) -> Result<()> {
    let snapshot = match fetch_manderqueue_messages(mq, run_id) {
        Ok(value) => value,
        Err(err) => {
            if mq.fail_closed {
                return Err(err);
            }
            json!({
                "schema_version": "gepa_manderqueue_inbox.v1",
                "ok": false,
                "error": err.to_string(),
                "messages": [],
            })
        }
    };
    write_json(&state_dir.join("manderqueue_inbox.json"), &snapshot)?;
    let guidance = guidance_markdown(&snapshot);
    let path = state_dir.join("guidance.md");
    fs::write(&path, guidance).map_err(|source| OptimizerError::io(&path, source))?;
    Ok(())
}

fn fetch_manderqueue_messages(mq: &ManderqueueConfig, run_id: &str) -> Result<Value> {
    let base = mq
        .base_url
        .as_deref()
        .unwrap_or("")
        .trim()
        .trim_end_matches('/');
    if base.is_empty() {
        return Ok(json!({
            "schema_version": "gepa_manderqueue_inbox.v1",
            "ok": true,
            "enabled": true,
            "base_url": Value::Null,
            "thread_id": mq.thread_id,
            "messages": [],
            "note": "manderqueue enabled without base_url; inbox is empty this turn",
        }));
    }
    let token = std::env::var(&mq.token_env).unwrap_or_default();
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(mq.poll_seconds.max(1)))
        .build()
        .map_err(|err| OptimizerError::Config(format!("manderqueue client: {err}")))?;
    let health = client.get(format!("{base}/health")).send();
    if let Err(err) = health {
        return Err(OptimizerError::Config(format!(
            "manderqueue health failed: {err}"
        )));
    }
    let Some(thread_id) = mq.thread_id.as_deref().filter(|id| !id.is_empty()) else {
        return Ok(json!({
            "schema_version": "gepa_manderqueue_inbox.v1",
            "ok": true,
            "base_url": base,
            "run_id": run_id,
            "messages": [],
            "note": "no thread_id yet; create one via POST /v1/threads then set gepa.operator.manderqueue.thread_id",
        }));
    };
    let mut req = client.get(format!("{base}/v1/threads/{thread_id}/messages"));
    if !token.is_empty() {
        req = req.header("authorization", format!("Bearer {token}"));
    }
    let response = req
        .query(&[("after_seq", "0"), ("limit", "50")])
        .send()
        .map_err(|err| OptimizerError::Config(format!("manderqueue read: {err}")))?;
    let status = response.status();
    let body = response
        .text()
        .map_err(|err| OptimizerError::Config(format!("manderqueue body: {err}")))?;
    if !status.is_success() {
        return Err(OptimizerError::Config(format!(
            "manderqueue {status}: {body}"
        )));
    }
    let messages: Value = serde_json::from_str(&body).unwrap_or_else(|_| json!([]));
    Ok(json!({
        "schema_version": "gepa_manderqueue_inbox.v1",
        "ok": true,
        "base_url": base,
        "thread_id": thread_id,
        "run_id": run_id,
        "messages": messages,
    }))
}

fn guidance_markdown(snapshot: &Value) -> String {
    let mut out = String::from("# Operator guidance (Manderqueue)\n\n");
    let messages = snapshot
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if messages.is_empty() {
        out.push_str("No new operator messages this turn.\n");
        if let Some(note) = snapshot.get("note").and_then(Value::as_str) {
            out.push('\n');
            out.push_str(note);
            out.push('\n');
        }
        return out;
    }
    for message in messages {
        let body = message
            .get("body")
            .and_then(Value::as_str)
            .unwrap_or_else(|| message.get("text").and_then(Value::as_str).unwrap_or(""));
        if body.is_empty() {
            continue;
        }
        out.push_str("- ");
        out.push_str(body);
        out.push('\n');
    }
    out
}

fn write_json(path: &PathBuf, value: &Value) -> Result<()> {
    let encoded = serde_json::to_vec_pretty(value)?;
    let mut file = fs::File::create(path).map_err(|source| OptimizerError::io(path, source))?;
    file.write_all(&encoded)
        .map_err(|source| OptimizerError::io(path, source))?;
    file.write_all(b"\n")
        .map_err(|source| OptimizerError::io(path, source))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use synth_optimizer_platform::{GepaConfig, RunConfig};

    fn config_with_operator(operator: GepaOperatorConfig) -> SynthOptimizerConfig {
        let mut config = SynthOptimizerConfig::default();
        config.run = RunConfig {
            run_id: "gepa_operator_test".to_string(),
            ..RunConfig::default()
        };
        config.gepa = GepaConfig {
            operator,
            ..GepaConfig::default()
        };
        config
    }

    #[test]
    fn disabled_operator_writes_nothing() {
        let dir = std::env::temp_dir().join(format!(
            "gepa-operator-off-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        prepare_operator_workspace(&SynthOptimizerConfig::default(), &dir).unwrap();
        assert!(!dir.join("state/operator.json").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn enabled_scratchpad_and_hypotheses_are_created_once() {
        let dir = std::env::temp_dir().join(format!(
            "gepa-operator-on-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let mut operator = GepaOperatorConfig::default();
        operator.scratchpad.enabled = true;
        operator.hypotheses.enabled = true;
        prepare_operator_workspace(&config_with_operator(operator), &dir).unwrap();
        let scratch = dir.join("state/scratchpad.md");
        let hypo = dir.join("state/hypotheses.json");
        assert!(scratch.exists());
        assert!(hypo.exists());
        fs::write(&scratch, "keep me\n").unwrap();
        let mut operator = GepaOperatorConfig::default();
        operator.scratchpad.enabled = true;
        operator.hypotheses.enabled = true;
        prepare_operator_workspace(&config_with_operator(operator), &dir).unwrap();
        assert_eq!(fs::read_to_string(&scratch).unwrap(), "keep me\n");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn manderqueue_without_url_writes_empty_inbox_when_not_fail_closed() {
        let dir = std::env::temp_dir().join(format!(
            "gepa-operator-mq-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let mut operator = GepaOperatorConfig::default();
        operator.manderqueue.enabled = true;
        operator.manderqueue.fail_closed = false;
        prepare_operator_workspace(&config_with_operator(operator), &dir).unwrap();
        let inbox: Value = serde_json::from_str(
            &fs::read_to_string(dir.join("state/manderqueue_inbox.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(inbox["ok"], true);
        assert!(dir.join("state/guidance.md").exists());
        let harvested = harvest_operator_workspace(&dir).unwrap();
        assert_eq!(harvested["scratchpad_exists"], false);
        assert_eq!(harvested["guidance_exists"], true);
        assert_eq!(harvested["manderqueue_inbox"]["ok"], true);
        let _ = fs::remove_dir_all(&dir);
    }
}
