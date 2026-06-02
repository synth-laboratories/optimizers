use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::CandidateRecord;
use reqwest::blocking::Client;
use serde_json::{json, Map, Value};
use synth_optimizer_platform::{
    run_turn, AgentTurnOutcome, CodexTurnRequest, OptimizerError, PromptProgram, Result,
    SynthOptimizerConfig,
};

const GEPA_REFLECTIVE_FRAME_SCHEMA_VERSION: &str = "gepa_reflective_frame.v1";
const CONTAINER_SENSOR_ADAPTER_ID: &str = "synth.container_sensor_frame_adapter";
const CONTAINER_SENSOR_ADAPTER_VERSION: &str = "v1";
const GEPA_ADAPTER_SOURCE: &str = "https://gepa-ai.github.io/gepa/guides/adapters/";
const GEPA_ALGORITHM_ID: &str = "synth_gepa.v1";
const GEPA_WORKSPACE_PROPOSAL_SCHEMA_VERSION: &str = "gepa_workspace_proposal_v3";

pub(crate) struct CodexProposerInput<'a> {
    pub config: &'a SynthOptimizerConfig,
    pub program: &'a PromptProgram,
    pub parent: &'a CandidateRecord,
    pub candidates: &'a [CandidateRecord],
    pub generation: usize,
    pub task_pool_rows: Value,
    pub workspace_dir: PathBuf,
}

pub(crate) struct CodexStalenessReviewerInput<'a> {
    pub config: &'a SynthOptimizerConfig,
    pub program: &'a PromptProgram,
    pub item: Value,
    pub stale_candidates: Vec<CandidateRecord>,
    pub current_best: Option<CandidateRecord>,
    pub pool_summary: Value,
    pub workspace_dir: PathBuf,
}

#[derive(Default)]
struct ProposerCandidateEvidenceStats {
    rollouts: usize,
    wins: usize,
    losses: usize,
    reward_sum: f64,
}

pub(crate) fn run_codex_app_server_proposer(input: CodexProposerInput<'_>) -> Result<Value> {
    materialize_workspace(&input)?;
    let model = input
        .config
        .proposer
        .model
        .clone()
        .unwrap_or_else(|| "gpt-5.4-mini".to_string());
    let timeout = Duration::from_secs(input.config.proposer.timeout_seconds.max(1));
    let outcome = run_turn(CodexTurnRequest {
        run_id: &input.config.run.run_id,
        proposer: &input.config.proposer,
        workspace_dir: &input.workspace_dir,
        model: &model,
        client_name: "synth-optimizers-gepa",
        client_title: "synth-optimizers GEPA",
        client_version: env!("CARGO_PKG_VERSION"),
        thread_start_params: thread_start_params(&input, &model),
        turn_start_params: turn_start_params(&input, &model)?,
        timeout,
    })?;
    build_response_from_outcome(&input, &model, outcome)
}

pub(crate) fn run_codex_staleness_reviewer(
    input: CodexStalenessReviewerInput<'_>,
) -> Result<Value> {
    if input.config.proposer.backend != "codex_app_server" {
        return Err(OptimizerError::Config(
            "gepa.pipeline.staleness_policy = reflective requires proposer.backend = \"codex_app_server\" for the staleness reviewer"
                .to_string(),
        ));
    }
    materialize_staleness_review_workspace(&input)?;
    let model = input
        .config
        .proposer
        .model
        .clone()
        .unwrap_or_else(|| "gpt-5.4-mini".to_string());
    let timeout = Duration::from_secs(input.config.proposer.timeout_seconds.max(1));
    let outcome = run_turn(CodexTurnRequest {
        run_id: &input.config.run.run_id,
        proposer: &input.config.proposer,
        workspace_dir: &input.workspace_dir,
        model: &model,
        client_name: "synth-optimizers-gepa",
        client_title: "synth-optimizers GEPA Staleness Reviewer",
        client_version: env!("CARGO_PKG_VERSION"),
        thread_start_params: staleness_thread_start_params(&input, &model),
        turn_start_params: staleness_turn_start_params(&input, &model),
        timeout,
    })?;
    build_staleness_review_response(&input, &model, outcome)
}

pub(crate) fn run_deepseek_chat_proposer(input: CodexProposerInput<'_>) -> Result<Value> {
    if !input
        .config
        .proposer
        .provider
        .eq_ignore_ascii_case("deepseek")
    {
        return Err(OptimizerError::Config(
            "proposer.backend = \"deepseek_chat\" requires proposer.provider = \"deepseek\""
                .to_string(),
        ));
    }
    materialize_workspace(&input)?;
    let model = input
        .config
        .proposer
        .model
        .clone()
        .unwrap_or_else(|| "deepseek-v4-flash".to_string());
    let api_key_env =
        non_empty(input.config.proposer.api_key_env.as_deref()).unwrap_or("DEEPSEEK_API_KEY");
    let api_key = env::var(api_key_env)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            OptimizerError::Proposer(format!(
                "proposer.backend = \"deepseek_chat\" requires non-empty {api_key_env}"
            ))
        })?;
    let base_url = input
        .config
        .proposer
        .base_url
        .as_deref()
        .unwrap_or("https://api.deepseek.com")
        .trim_end_matches('/')
        .to_string();
    let request = json!({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are the GEPA workspace proposer. Return only a JSON object that matches the requested manifest schema."
            },
            {
                "role": "user",
                "content": deepseek_chat_prompt(&input)?
            }
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        // Manifests for large minibatches can be long; 4096 truncates the JSON
        // mid-string ("EOF while parsing"). deepseek-v4-flash supports far more.
        "max_tokens": 32768,
        "stream": false
    });
    let client = Client::builder()
        .timeout(Duration::from_secs(
            input.config.proposer.timeout_seconds.max(1),
        ))
        .build()?;
    // Resilience: retry the whole request when the response can't be decoded,
    // the content is missing, or the model's manifest JSON is malformed/truncated
    // (transient DeepSeek errors). HTTP 429/5xx are already retried inside
    // post_deepseek_chat_completion; this outer loop also covers decode/parse.
    let (chat_response, manifest) = {
        let mut last_err: Option<String> = None;
        let mut result: Option<(Value, Value)> = None;
        for attempt in 1..=3usize {
            match post_deepseek_chat_completion(&client, &base_url, &api_key, &request, 4) {
                Ok(resp) => {
                    match resp
                        .pointer("/choices/0/message/content")
                        .and_then(Value::as_str)
                    {
                        Some(content) => match serde_json::from_str::<Value>(content.trim()) {
                            Ok(m) => {
                                result = Some((resp, m));
                                break;
                            }
                            Err(e) => last_err = Some(format!("manifest JSON parse failed: {e}")),
                        },
                        None => {
                            last_err =
                                Some("response missing choices[0].message.content".to_string())
                        }
                    }
                }
                Err(e) => last_err = Some(e.to_string()),
            }
            if attempt < 3 {
                std::thread::sleep(Duration::from_secs(2 * attempt as u64));
            }
        }
        match result {
            Some(pair) => pair,
            None => {
                return Err(OptimizerError::Proposer(format!(
                    "DeepSeek chat proposer failed after 3 attempts: {}",
                    last_err.unwrap_or_else(|| "unknown error".to_string())
                )))
            }
        }
    };
    let manifest_path = input.workspace_dir.join("proposal").join("manifest.json");
    write_json(&manifest_path, &manifest)?;
    let manifest = read_manifest(&input.workspace_dir)?;
    let proposals = proposals_from_manifest(&manifest)?;
    let evidence_warnings = manifest_evidence_warnings(&input, &manifest, &proposals);
    let usage = chat_response.get("usage").cloned().ok_or_else(|| {
        OptimizerError::Proposer("DeepSeek chat proposer response missing usage".to_string())
    })?;
    write_deepseek_chat_artifacts(&input, &request, &chat_response)?;
    write_workspace_pack_manifest(&input.workspace_dir)?;
    Ok(json!({
        "backend": "deepseek_chat",
        "runtime_substrate": "local",
        "workspace": input.workspace_dir,
        "manifest": manifest,
        "proposals": proposals,
        "usage": usage,
        "evidence_warnings": evidence_warnings,
    }))
}

fn deepseek_chat_prompt(input: &CodexProposerInput<'_>) -> Result<String> {
    let best_practices = resolved_prompting_best_practices(input)?;
    let mut prompt = String::new();
    prompt.push_str(&proposer_prompt_context(input));
    prompt.push_str("\n\nPrompting best practices:\n");
    prompt.push_str(best_practices.trim());
    prompt.push_str("\n\nManifest schema:\n");
    prompt.push_str(&proposal_schema(input));
    prompt.push_str("\n\nUse these workspace files as the complete evidence packet.\n");
    for path in [
        "state/proposer_metadata.json",
        "state/task_info.json",
        "state/program_contract.json",
        "state/parent_payload.json",
        "state/candidate_deltas.json",
        "state/proposer_failure_summary.json",
        "state/proposer_repair_hints.json",
        "state/proposer_examples.json",
        "state/rollouts.json",
        "state/scores.json",
        "state/proposal_request.json",
    ] {
        let file_path = input.workspace_dir.join(path);
        let text = fs::read_to_string(&file_path)
            .map_err(|source| OptimizerError::io(&file_path, source))?;
        // Bound each evidence file so the prompt stays within the model context
        // window. With large minibatches + accepted candidates, rollouts.json /
        // scores.json grow past 1M tokens otherwise (DeepSeek 400: context length).
        const MAX_EVIDENCE_CHARS: usize = 120_000;
        let text = if text.chars().count() > MAX_EVIDENCE_CHARS {
            let head: String = text.chars().take(MAX_EVIDENCE_CHARS).collect();
            format!("{head}\n…[truncated to {MAX_EVIDENCE_CHARS} chars to fit context budget]…")
        } else {
            text
        };
        prompt.push_str("\n\n--- ");
        prompt.push_str(path);
        prompt.push_str(" ---\n");
        prompt.push_str(&text);
    }
    prompt.push_str(
        "\n\nReturn strict JSON only. Do not wrap it in Markdown. The JSON object must have \
         schema_version, critique, evidence, rationale, and proposals. Each proposed_payload \
         must contain full replacement text for every target module.",
    );
    Ok(prompt)
}

fn post_deepseek_chat_completion(
    client: &Client,
    base_url: &str,
    api_key: &str,
    request: &Value,
    max_attempts: usize,
) -> Result<Value> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    for attempt in 1..=max_attempts {
        let response = client
            .post(&url)
            .bearer_auth(api_key)
            .json(request)
            .send()?;
        let status = response.status();
        let text = response.text()?;
        if status.is_success() {
            return Ok(serde_json::from_str(&text)?);
        }
        if attempt < max_attempts && matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504) {
            let delay = match attempt {
                1 => Duration::from_secs(2),
                2 => Duration::from_secs(5),
                _ => Duration::from_secs(10),
            };
            std::thread::sleep(delay);
            continue;
        }
        return Err(OptimizerError::Proposer(format!(
            "DeepSeek chat proposer failed after {attempt}/{max_attempts} attempts with status \
             {}: {}",
            status,
            text.chars().take(1000).collect::<String>()
        )));
    }
    Err(OptimizerError::Proposer(
        "DeepSeek chat proposer retry loop exited unexpectedly".to_string(),
    ))
}

fn write_deepseek_chat_artifacts(
    input: &CodexProposerInput<'_>,
    request: &Value,
    response: &Value,
) -> Result<()> {
    let artifact_dir = input.workspace_dir.join(".agent_artifacts");
    fs::create_dir_all(&artifact_dir)
        .map_err(|source| OptimizerError::io(&artifact_dir, source))?;
    write_json(
        &artifact_dir.join("deepseek_chat_request.json"),
        &json!({
            "model": request.get("model"),
            "message_count": request.get("messages").and_then(Value::as_array).map(Vec::len),
            "max_tokens": request.get("max_tokens"),
            "response_format": request.get("response_format"),
            "thinking": request.get("thinking"),
        }),
    )?;
    write_json(&artifact_dir.join("deepseek_chat_response.json"), response)
}

fn build_response_from_outcome(
    input: &CodexProposerInput<'_>,
    model: &str,
    outcome: AgentTurnOutcome,
) -> Result<Value> {
    let usage = outcome
        .usage
        .clone()
        .unwrap_or_else(|| json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}));
    let mut prevalidation_response = json!({
        "backend": "codex_app_server",
        "runtime_substrate": input.config.proposer.runtime_substrate.as_str(),
        "workspace": input.workspace_dir,
        "usage": usage.clone(),
        "manifest_validation": "pending",
    });
    if let Some(receipt) = outcome.supervisor_receipt.as_ref() {
        prevalidation_response["supervisor_receipt"] = serde_json::to_value(receipt)?;
    }
    if let Some(shutdown_warning) = outcome.shutdown_warning.as_ref() {
        prevalidation_response["shutdown_warning"] = Value::String(shutdown_warning.clone());
    }
    write_agent_artifacts(
        input,
        model,
        &outcome.thread_id,
        &outcome.turn_id,
        &outcome.thread_response,
        &outcome.final_turn,
        &prevalidation_response,
        &outcome,
    )?;
    let manifest = read_manifest(&input.workspace_dir)?;
    let proposals = proposals_from_manifest(&manifest)?;
    let mut evidence_warnings = manifest_evidence_warnings(input, &manifest, &proposals);
    if outcome.usage.is_none() {
        evidence_warnings.push(
            "proposer usage missing from codex turn payload; token counts may be incomplete"
                .to_string(),
        );
    }
    let mut response = json!({
        "backend": "codex_app_server",
        "runtime_substrate": input.config.proposer.runtime_substrate.as_str(),
        "workspace": input.workspace_dir,
        "manifest": manifest,
        "proposals": proposals,
        "usage": usage,
        "evidence_warnings": evidence_warnings,
    });
    if let Some(receipt) = outcome.supervisor_receipt.as_ref() {
        response["supervisor_receipt"] = serde_json::to_value(receipt)?;
    }
    if let Some(shutdown_warning) = outcome.shutdown_warning.as_ref() {
        response["shutdown_warning"] = Value::String(shutdown_warning.clone());
    }
    write_agent_artifacts(
        input,
        model,
        &outcome.thread_id,
        &outcome.turn_id,
        &outcome.thread_response,
        &outcome.final_turn,
        &response,
        &outcome,
    )?;
    write_workspace_pack_manifest(&input.workspace_dir)?;
    Ok(response)
}

fn build_staleness_review_response(
    input: &CodexStalenessReviewerInput<'_>,
    model: &str,
    outcome: AgentTurnOutcome,
) -> Result<Value> {
    let usage = outcome
        .usage
        .clone()
        .unwrap_or_else(|| json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}));
    let verdict_path = input.workspace_dir.join("review").join("verdict.json");
    let verdict = read_staleness_verdict_json(&verdict_path)?;
    validate_staleness_verdict(&verdict)?;
    write_json(&verdict_path, &verdict)?;
    let mut response = json!({
        "backend": "codex_app_server",
        "runtime_substrate": input.config.proposer.runtime_substrate.as_str(),
        "workspace": input.workspace_dir,
        "model": model,
        "usage": usage,
        "verdict": verdict,
    });
    if let Some(receipt) = outcome.supervisor_receipt.as_ref() {
        response["supervisor_receipt"] = serde_json::to_value(receipt)?;
    }
    if let Some(shutdown_warning) = outcome.shutdown_warning.as_ref() {
        response["shutdown_warning"] = Value::String(shutdown_warning.clone());
    }
    write_staleness_agent_artifacts(input, model, &outcome, &response)?;
    write_workspace_pack_manifest(&input.workspace_dir)?;
    Ok(response)
}

fn validate_staleness_verdict(verdict: &Value) -> Result<()> {
    let verdict_text = verdict
        .get("verdict")
        .or_else(|| verdict.get("decision"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if !matches!(
        verdict_text.as_str(),
        "accept" | "accept_as_is" | "discard" | "drop" | "patch" | "repair"
    ) {
        return Err(OptimizerError::Proposer(format!(
            "staleness reviewer verdict must be accept, discard, or patch; got {verdict_text:?}"
        )));
    }
    if matches!(verdict_text.as_str(), "patch" | "repair")
        && !verdict
            .get("patched_payload")
            .or_else(|| verdict.get("payload"))
            .is_some_and(Value::is_object)
    {
        return Err(OptimizerError::Proposer(
            "staleness reviewer patch verdict must include patched_payload".to_string(),
        ));
    }
    Ok(())
}

fn write_staleness_agent_artifacts(
    input: &CodexStalenessReviewerInput<'_>,
    model: &str,
    outcome: &AgentTurnOutcome,
    response: &Value,
) -> Result<()> {
    let artifact_dir = input.workspace_dir.join(".agent_artifacts");
    fs::create_dir_all(&artifact_dir)
        .map_err(|source| OptimizerError::io(&artifact_dir, source))?;
    write_json(
        &artifact_dir.join("opencode_session.json"),
        &json!({
            "schema_version": "gepa_codex_staleness_review_session.v1",
            "backend": "codex_app_server",
            "model": model,
            "thread_id": outcome.thread_id,
            "turn_id": outcome.turn_id,
            "workspace": input.workspace_dir,
            "supervisor_receipt": outcome.supervisor_receipt,
            "thread_response": outcome.thread_response,
            "final_turn": outcome.final_turn,
        }),
    )?;
    write_json(
        &artifact_dir.join("opencode_messages.json"),
        &json!({
            "schema_version": "gepa_codex_staleness_review_messages.v1",
            "sent": outcome.sent_messages,
            "received": outcome.received_messages,
        }),
    )?;
    write_json(&artifact_dir.join("opencode_response.json"), response)?;
    let mut events = String::new();
    for message in &outcome.received_messages {
        events.push_str(&serde_json::to_string(message)?);
        events.push('\n');
    }
    write_text(&artifact_dir.join("opencode_sse_events.jsonl"), &events)?;
    Ok(())
}

fn materialize_staleness_review_workspace(input: &CodexStalenessReviewerInput<'_>) -> Result<()> {
    let state_dir = input.workspace_dir.join("state");
    let review_dir = input.workspace_dir.join("review");
    fs::create_dir_all(&state_dir).map_err(|source| OptimizerError::io(&state_dir, source))?;
    fs::create_dir_all(&review_dir).map_err(|source| OptimizerError::io(&review_dir, source))?;
    write_text(
        &input.workspace_dir.join("README.md"),
        &staleness_review_readme(input),
    )?;
    write_text(
        &review_dir.join("VERDICT_SCHEMA.md"),
        &staleness_verdict_schema(input),
    )?;
    write_json(
        &review_dir.join("verdict.json"),
        &json!({
            "schema_version": "gepa_staleness_review_v1",
            "verdict": "accept",
            "reason": "",
            "patched_payload": null,
        }),
    )?;
    write_json(
        &state_dir.join("staleness_review_request.json"),
        &json!({
            "schema_version": "gepa_staleness_review_request_v1",
            "run_id": input.config.run.run_id,
            "target_modules": input.config.candidate.target_modules,
            "item": input.item,
            "stale_candidates": input.stale_candidates,
            "current_best": input.current_best,
            "pool_summary": input.pool_summary,
            "program": input.program,
        }),
    )?;
    write_workspace_pack_manifest(&input.workspace_dir)?;
    Ok(())
}

fn staleness_review_readme(input: &CodexStalenessReviewerInput<'_>) -> String {
    format!(
        r#"# GEPA Staleness Reviewer Workspace

You are reviewing stale FlashEvolve work before it is folded back into the current artifact pool.

Read `state/staleness_review_request.json` and `review/VERDICT_SCHEMA.md`.
Choose exactly one verdict:

- `accept`: the stale candidate is still compatible with the current pool.
- `discard`: the stale candidate is obsolete or conflicts with newer accepted pool changes.
- `patch`: provide a full replacement `patched_payload` for the target modules.

If patching, preserve every target module key and write a complete payload object. Target modules: {}.
Write strict JSON to `review/verdict.json`.
"#,
        input.config.candidate.target_modules.join(", ")
    )
}

fn staleness_verdict_schema(input: &CodexStalenessReviewerInput<'_>) -> String {
    let payload_hint = if input.config.candidate.target_modules.len() > 1 {
        "When verdict is patch, patched_payload must include every target module key."
    } else {
        "When verdict is patch, patched_payload must include the single target module key."
    };
    format!(
        r#"# GEPA Staleness Review Verdict Schema

Write `review/verdict.json` as strict JSON:

```json
{{
  "schema_version": "gepa_staleness_review_v1",
  "verdict": "accept | discard | patch",
  "reason": "Short explanation grounded in state/staleness_review_request.json.",
  "patched_payload": {{
    "<target_module>": "<full replacement instruction>"
  }}
}}
```

Rules:

- Use `accept` when the stale candidate's prompt remains compatible with the current pool.
- Use `discard` when the candidate duplicates, conflicts with, or is dominated by current pool changes.
- Use `patch` when a small language-space repair can rebase the stale candidate onto the current pool.
- {payload_hint}
- If verdict is not `patch`, set `patched_payload` to null.
"#
    )
}

fn materialize_workspace(input: &CodexProposerInput<'_>) -> Result<()> {
    let state_dir = input.workspace_dir.join("state");
    let proposal_dir = input.workspace_dir.join("proposal");
    fs::create_dir_all(&state_dir).map_err(|source| OptimizerError::io(&state_dir, source))?;
    fs::create_dir_all(&proposal_dir)
        .map_err(|source| OptimizerError::io(&proposal_dir, source))?;

    write_text(
        &input.workspace_dir.join("README.md"),
        &workspace_readme(input),
    )?;
    let prompting_best_practices = resolved_prompting_best_practices(input)?;
    write_text(
        &input.workspace_dir.join("prompting_best_practices.md"),
        &prompting_best_practices,
    )?;
    write_text(
        &proposal_dir.join("PROPOSAL_SCHEMA.md"),
        &proposal_schema(input),
    )?;
    write_json(
        &proposal_dir.join("manifest.json"),
        &json!({
            "schema_version": GEPA_WORKSPACE_PROPOSAL_SCHEMA_VERSION,
            "critique": "",
            "evidence": {
                "reviewed_files": [],
                "candidate_comparison": "",
                "failure_patterns": [],
                "winning_patterns": [],
                "example_ids_used": [],
            },
            "rationale": "",
            "proposals": [],
        }),
    )?;
    let parent_payload = json!(&input.parent.payload);
    let proposal_request = proposal_request(input, &prompting_best_practices);
    let candidates = candidates_read_model(input);
    let candidate_deltas = candidate_deltas_read_model(input);
    let rollouts = rollouts_read_model(input);
    let proposer_examples = proposer_examples_read_model(input);
    let proposer_failure_summary = proposer_failure_summary_read_model(input, &proposer_examples);
    let proposer_repair_hints = proposer_repair_hints_read_model(input, &proposer_examples);
    let proposer_metadata = proposer_metadata_read_model(
        input,
        &rollouts,
        &proposer_examples,
        &proposer_failure_summary,
    );
    let proposer_readme = proposer_readme_read_model();
    let scores = scores_read_model(input);
    let evidence_frames = evidence_frames_read_model(input);
    let reflective_frames = reflective_frames_read_model(input);
    let links = links_read_model(input);
    let pareto_front = pareto_front_read_model(input);
    let gepa_summary = gepa_summary_read_model(input, &rollouts);
    let candidate_selector = candidate_selector_read_model(input);
    let batch_sampler = batch_sampler_read_model(input);
    let acceptance = acceptance_read_model(input);
    let task_pools = task_pools_read_model(input);
    let algorithm_read_model = json!({
        "schema_version": "gepa_algorithm_read_model_v1",
        "generation": input.generation,
        "parent_candidate_id": input.parent.candidate_id,
        "target_modules": input.config.candidate.target_modules,
        "proposals_per_round": input.config.gepa.proposals_per_generation,
        "candidate_selector": candidate_selector,
        "batch_sampler": batch_sampler,
        "acceptance": acceptance.clone(),
        "task_pools": task_pools,
        "reflection_examples": reflection_examples_read_model(input),
        "parent_payload": parent_payload,
        "candidates": candidates,
        "candidate_deltas": candidate_deltas,
        "rollouts": rollouts,
        "proposer_examples": proposer_examples,
        "proposer_failure_summary": proposer_failure_summary,
        "proposer_repair_hints": proposer_repair_hints,
        "proposer_metadata": proposer_metadata,
        "proposer_readme": proposer_readme,
        "scores": scores,
        "evidence_frames": evidence_frames,
        "reflective_frames": reflective_frames,
        "links": links,
        "pareto_front": pareto_front,
        "proposal_request": proposal_request,
        "summary": gepa_summary,
    });
    write_json(
        &state_dir.join("run_context.json"),
        &json!({
            "run_id": input.config.run.run_id,
            "generation": input.generation,
            "task": "GEPA prompt proposal",
            "program_id": input.program.program_id,
            "target_modules": input.config.candidate.target_modules,
            "proposals_per_generation": input.config.gepa.proposals_per_generation,
            "proposals_per_round": input.config.gepa.proposals_per_generation,
            "parent_candidate_id": input.parent.candidate_id,
            "acceptance": acceptance,
            "task_pool_counts": task_pool_counts(input),
        }),
    )?;
    write_json(
        &state_dir.join("task_info.json"),
        &task_info_value(input).cloned().unwrap_or(Value::Null),
    )?;
    write_json(
        &state_dir.join("program_contract.json"),
        &json!({
            "program_id": input.program.program_id,
            "target_modules": input.config.candidate.target_modules,
            "mutable_fields": input.program.mutable_field_ids(),
            "program": input.program,
        }),
    )?;
    write_json(
        &state_dir.join("program.json"),
        &serde_json::to_value(input.program)?,
    )?;
    write_json(
        &state_dir.join("parent_candidate.json"),
        &serde_json::to_value(input.parent)?,
    )?;
    write_json(&state_dir.join("parent_payload.json"), &parent_payload)?;
    write_json(
        &state_dir.join("candidates.json"),
        &candidates_read_model(input),
    )?;
    write_json(
        &state_dir.join("candidate_deltas.json"),
        &candidate_deltas_read_model(input),
    )?;
    write_json(
        &state_dir.join("rollouts.json"),
        &rollouts_read_model(input),
    )?;
    write_json(
        &state_dir.join("proposer_examples.json"),
        &proposer_examples_read_model(input),
    )?;
    write_json(
        &state_dir.join("proposer_failure_summary.json"),
        &proposer_failure_summary,
    )?;
    write_json(
        &state_dir.join("proposer_repair_hints.json"),
        &proposer_repair_hints,
    )?;
    write_json(
        &state_dir.join("proposer_metadata.json"),
        &proposer_metadata,
    )?;
    write_json(&state_dir.join("proposer_readme.json"), &proposer_readme)?;
    write_json(&state_dir.join("scores.json"), &scores_read_model(input))?;
    write_json(
        &state_dir.join("evidence_frames.json"),
        &evidence_frames_read_model(input),
    )?;
    write_json(
        &state_dir.join("reflective_frames.json"),
        &reflective_frames_read_model(input),
    )?;
    write_json(&state_dir.join("links.json"), &links_read_model(input))?;
    write_json(
        &state_dir.join("task_pools.json"),
        &task_pools_read_model(input),
    )?;
    write_json(
        &state_dir.join("algorithm_read_model.json"),
        &algorithm_read_model,
    )?;
    write_json(
        &state_dir.join("pareto_front.json"),
        &pareto_front_read_model(input),
    )?;
    write_json(&state_dir.join("gepa_sidecar.json"), &algorithm_read_model)?;
    write_json(&state_dir.join("gepa_summary.json"), &gepa_summary)?;
    write_json(&state_dir.join("proposal_request.json"), &proposal_request)?;
    write_json(
        &state_dir.join("reflector_input.json"),
        &reflector_input_read_model(input, &prompting_best_practices),
    )?;
    write_workspace_pack_manifest(&input.workspace_dir)?;
    Ok(())
}

fn resolved_prompting_best_practices(input: &CodexProposerInput<'_>) -> Result<String> {
    let prompt = &input.config.proposer.prompt;
    if let Some(text) = prompt.best_practices.as_deref() {
        return Ok(text.to_string());
    }
    if let Some(path) = &prompt.best_practices_path {
        return fs::read_to_string(path).map_err(|source| OptimizerError::io(path, source));
    }
    Ok(crate::default_proposer_best_practices().to_string())
}

fn write_workspace_pack_manifest(workspace_dir: &Path) -> Result<()> {
    let state_dir = workspace_dir.join("state");
    fs::create_dir_all(&state_dir).map_err(|source| OptimizerError::io(&state_dir, source))?;
    let mut files = Vec::new();
    collect_workspace_files(workspace_dir, workspace_dir, &mut files)?;
    files.sort_by(|left, right| {
        left.get("path")
            .and_then(Value::as_str)
            .cmp(&right.get("path").and_then(Value::as_str))
    });
    write_json(
        &state_dir.join("workspace_pack_manifest.json"),
        &json!({
            "schema_version": "gepa_workspace_pack_manifest.v1",
            "file_count": files.len(),
            "files": files,
        }),
    )
}

fn collect_workspace_files(root: &Path, current: &Path, files: &mut Vec<Value>) -> Result<()> {
    for entry in fs::read_dir(current).map_err(|source| OptimizerError::io(current, source))? {
        let entry = entry.map_err(|source| OptimizerError::io(current, source))?;
        let path = entry.path();
        let relative = path.strip_prefix(root).unwrap_or(&path);
        if should_skip_workspace_manifest_path(relative) {
            continue;
        }
        let metadata = entry
            .metadata()
            .map_err(|source| OptimizerError::io(&path, source))?;
        if metadata.is_dir() {
            collect_workspace_files(root, &path, files)?;
        } else if metadata.is_file() {
            files.push(json!({
                "path": relative.to_string_lossy(),
                "bytes": metadata.len(),
            }));
        }
    }
    Ok(())
}

fn should_skip_workspace_manifest_path(path: &Path) -> bool {
    path.components().any(|component| {
        let text = component.as_os_str().to_string_lossy();
        matches!(text.as_ref(), ".codex_home" | ".codex_api_key_home")
    })
}

#[allow(clippy::too_many_arguments)]
fn write_agent_artifacts(
    input: &CodexProposerInput<'_>,
    model: &str,
    thread_id: &str,
    turn_id: &str,
    thread_response: &Value,
    final_turn: &Value,
    response: &Value,
    outcome: &AgentTurnOutcome,
) -> Result<()> {
    let artifact_dir = input.workspace_dir.join(".agent_artifacts");
    fs::create_dir_all(&artifact_dir)
        .map_err(|source| OptimizerError::io(&artifact_dir, source))?;
    write_json(
        &artifact_dir.join("opencode_session.json"),
        &json!({
            "schema_version": "gepa_codex_app_server_session.v1",
            "backend": "codex_app_server",
            "model": model,
            "thread_id": thread_id,
            "turn_id": turn_id,
            "workspace": input.workspace_dir,
            "sandbox_mode": input.config.proposer.sandbox_mode,
            "approval_policy": input.config.proposer.approval_policy,
            "auth_mode": input.config.proposer.auth_mode,
            "runtime_substrate": input.config.proposer.runtime_substrate.as_str(),
            "supervisor_receipt": &outcome.supervisor_receipt,
            "thread_response": thread_response,
            "final_turn": final_turn,
        }),
    )?;
    write_json(
        &artifact_dir.join("opencode_messages.json"),
        &json!({
            "schema_version": "gepa_codex_app_server_messages.v1",
            "sent": &outcome.sent_messages,
            "received": &outcome.received_messages,
        }),
    )?;
    write_json(&artifact_dir.join("opencode_response.json"), response)?;
    let mut events = String::new();
    for message in &outcome.received_messages {
        events.push_str(&serde_json::to_string(message)?);
        events.push('\n');
    }
    write_text(&artifact_dir.join("opencode_sse_events.jsonl"), &events)?;
    Ok(())
}

fn workspace_readme(input: &CodexProposerInput<'_>) -> String {
    let proposal_policy = proposer_policy_text(input);
    format!(
        r#"# GEPA Proposer Workspace

You are proposing the next GEPA prompt candidate.

Read:

1. `prompting_best_practices.md` for the shared premise/context/task_priority/heuristics/constraints/rules typology.
2. `proposal/PROPOSAL_SCHEMA.md` for the exact manifest schema.
3. `state/proposer_metadata.json` for run/generation metadata, model names, target levers, counts, budgets, and top failures.
4. `state/proposer_readme.json` for a machine-readable file index.
5. `state/proposer_failure_summary.json` first for flat losses, wins, label confusions, text, expected labels, predictions, rewards, and prompt payloads.
6. `state/proposer_repair_hints.json` for generalized reflection hints, label-confusion clusters, and guard wins.
7. `state/proposer_examples.json` for every flat rollout evidence row.
8. `state/run_context.json` for the optimizer run context and target modules.
9. `state/task_info.json` for the container-declared task, output space, metrics, and proposer hints.
10. `state/program_contract.json` for the program and mutable fields.
11. `state/candidates.json` for candidate payloads and train/minibatch/heldout scores.
12. `state/candidate_deltas.json` for payload differences from the selected parent.
13. `state/rollouts.json` and `state/scores.json` for per-example rollouts and score summaries. Sensor-backed rows in `state/rollouts.json` include summaries, outcomes, expected outputs, predictions, text, rationale, and trace refs.
14. `state/evidence_frames.json`, `state/reflective_frames.json`, and `state/links.json` for durable nested rollout evidence. `state/reflective_frames.json` is an object; inspect `.frames[]`.
15. `state/task_pools.json` for pareto-eval, minibatch, reflection, and validation row pools.
16. `state/algorithm_read_model.json` for the complete GEPA read model.
17. `state/pareto_front.json`, `state/gepa_sidecar.json`, and `state/gepa_summary.json` for GEPA-specific mirrors.
18. `state/parent_payload.json` and `state/reflector_input.json` for the parent prompt and sampled wins/losses.

Before writing the manifest, inspect those files with shell, Python, or JQ and form a short evidence summary. Use `state/task_info.json`, rollout traces, rationales, and expected/predicted outputs to infer what kind of task this is before deciding what style of prompt edit is valid.
Use a real review workflow: summarize candidate scores and payloads, inspect Pareto membership, inspect rollout wins/losses, inspect the parent payload, then write `proposal/manifest.json`.

Reflect over the evidence like GEPA's Python workspace proposer. You have wide latitude over the prompt content: rewrite structure, add role priming, include numbered sections, restate the task contract, and add examples when the task policy allows them.

{proposal_policy}

Write exactly {proposal_count} distinct candidate proposals to `proposal/manifest.json`.
"#,
        proposal_policy = proposal_policy,
        proposal_count = input.config.gepa.proposals_per_generation
    )
}

fn proposal_schema(input: &CodexProposerInput<'_>) -> String {
    let payload_rule = if input.config.candidate.target_modules.len() > 1 {
        "- Multiple target modules are active. Each `proposed_payload` must be the full candidate payload, with every target module present and non-empty.\n"
    } else {
        "- Keep the change targeted to the target module named in `state/run_context.json`.\n"
    };
    let proposal_policy = proposer_policy_text(input);
    format!(
        r#"# GEPA Workspace Proposer Schema

Write `proposal/manifest.json` as strict JSON using this schema:

```json
{{
  "schema_version": "gepa_workspace_proposal_v3",
  "critique": "What the parent candidate is missing, grounded in state/ evidence.",
  "evidence": {{
    "reviewed_files": [
      "prompting_best_practices.md",
      "state/proposer_metadata.json",
      "state/proposer_readme.json",
      "state/run_context.json",
      "state/task_info.json",
      "state/program_contract.json",
      "state/algorithm_read_model.json",
      "state/candidates.json",
      "state/candidate_deltas.json",
      "state/proposer_failure_summary.json",
      "state/proposer_repair_hints.json",
      "state/proposer_examples.json",
      "state/rollouts.json",
      "state/scores.json",
      "state/evidence_frames.json",
      "state/reflective_frames.json",
      "state/task_pools.json",
      "state/links.json"
    ],
    "candidate_comparison": "Short comparison of parent, Pareto members, and recent candidates.",
    "failure_patterns": ["Observed failure pattern grounded in losing rollout examples."],
    "winning_patterns": ["Observed winning pattern grounded in successful rollout examples."],
    "example_ids_used": ["train:1", "train:10", "train:14"]
  }},
  "rationale": "Why the proposed prompts should improve the target module.",
  "proposals": [
    {{
      "proposal_type": "frontier_variation",
      "parent_candidate_ids": ["<pareto_candidate_id>"],
      "rationale": "Why this variation should help.",
      "proposed_payload": {{
        "<target_module>": "<full replacement instruction>"
      }}
    }},
    {{
      "proposal_type": "frontier_merge",
      "parent_candidate_ids": ["<pareto_candidate_id_1>", "<pareto_candidate_id_2>"],
      "rationale": "Which strengths this merge attempts to combine.",
      "proposed_payload": {{
        "<target_module>": "<full replacement instruction>"
      }}
    }}
  ]
}}
```

Rules:

- Read `prompting_best_practices.md`, `state/proposer_metadata.json`, `state/proposer_readme.json`, `state/run_context.json`, `state/task_info.json`, `state/program_contract.json`, `state/algorithm_read_model.json`, `state/candidates.json`, `state/candidate_deltas.json`, `state/proposer_failure_summary.json`, `state/proposer_repair_hints.json`, `state/proposer_examples.json`, `state/rollouts.json`, `state/scores.json`, `state/evidence_frames.json`, `state/reflective_frames.json`, `state/links.json`, `state/parent_payload.json`, and `state/reflector_input.json`.
- Preserve the exact top-level and evidence field names from the JSON schema. In particular, use `evidence.reviewed_files` and `evidence.example_ids_used`; do not rename them to `files_reviewed`, `example_ids`, or any other alias.
- Use shell/Python/JQ inspection to summarize the workspace before writing the manifest. Do not jump straight to editing `proposal/manifest.json`.
- Minimum review workflow: inspect `state/proposer_metadata.json`, inspect `state/task_info.json`, inspect candidate scores/payloads, inspect Pareto membership, inspect rollout wins/losses and trace refs, inspect parent payload, then write the manifest.
- Use `state/proposer_failure_summary.json`, `state/proposer_repair_hints.json`, and `state/proposer_examples.json` as the primary source for rollout rewards, failures, wins, expected outputs, predictions, and example text. Use nested evidence frames when task semantics or trace-level behavior are unclear.
- Use `prompting_best_practices.md` to classify each proposed change as a premise, context, task_priority, core_task_description, heuristic, constraint, rule, input_description, or output_description.
- Fill `evidence` with concrete files reviewed, candidate comparison, failure patterns, winning patterns, and example ids from `state/proposer_failure_summary.json`.
- Proposals should aim to generalize. Add structural sections (role, task, output rules, examples) and domain-specific rules only when they are task-valid.
- {proposal_policy}
- At most one proposal may be conservative. The remaining proposals must be very ambitious, high-variance, task-specific updates that could plausibly produce substantially better task performance than the parent, and each rationale must name the failure clusters it attacks.
- Shoot for large wins. Mild parent clarifications are wasted candidate budget unless they are the single conservative control.
- Do not waste candidates on generic output-contract polish, canonical-label reminders, or baseline paraphrases unless the dominant failures are actually output-format failures.
- Use whatever combination works: label-disambiguation rules, output-format constraints, structural rewrites, few-shot examples, role priming, edge-case enumeration. Distinct proposals should explore distinct strategies, not paraphrase each other.
- Create exactly `state/proposal_request.json.proposals_per_round` distinct proposals.
- Use `proposal_type="frontier_variation"` for a mutation of one Pareto-front candidate.
- Use `proposal_type="frontier_merge"` for an attempted combination of two Pareto-front candidates with complementary wins. If fewer than two Pareto-front candidates exist, replace requested merges with additional frontier variations.
- Do not propose a duplicate of an existing payload in `state/candidates.json`.
- Preserve all parent payload keys unless a key is intentionally changed.
- Each `proposed_payload` must be the full payload object to register as a GEPA candidate.
- For each proposal, at least one targeted module must change from the selected parent payload.
{payload_rule}"#
    )
}

fn proposal_request(input: &CodexProposerInput<'_>, prompting_best_practices: &str) -> Value {
    let proposal_count = input.config.gepa.proposals_per_generation;
    let pareto_front = compute_pareto_front(input);
    let members = sorted_pareto_member_ids(input, &pareto_front);
    let merge_count = if members.len() >= 2 {
        proposal_count / 3
    } else {
        0
    };
    let merge_pairs = merge_candidate_pairs(&members);
    let merge_common_ancestors = merge_pairs
        .iter()
        .map(|pair| {
            (
                pair.join("+"),
                common_ancestor_id(input, &[pair[0].clone(), pair[1].clone()]),
            )
        })
        .collect::<BTreeMap<_, _>>();
    json!({
        "proposal_count": proposal_count,
        "proposals_per_round": proposal_count,
        "frontier_variations": proposal_count.saturating_sub(merge_count),
        "frontier_merges": merge_count,
        "variation_parent_candidate_ids": members,
        "merge_candidate_pairs": merge_pairs,
        "merge_common_ancestors": merge_common_ancestors,
        "frontier_cells": pareto_front.cells.iter().take(200).cloned().collect::<Vec<_>>(),
        "frontier_type": pareto_front.frontier_type,
        "target_modules": input.config.candidate.target_modules,
        "parent_candidate_id": input.parent.candidate_id,
        "candidate_selector": candidate_selector_read_model(input),
        "batch_sampler": batch_sampler_read_model(input),
        "acceptance": acceptance_read_model(input),
        "task_pool_counts": task_pool_counts(input),
        "literal_example_policy": proposer_literal_policy_json(input),
        "prompting_best_practices": prompting_best_practices,
        "ambition_contract": [
            "At most one proposal may be conservative.",
            "Every other proposal must be a very ambitious, task-specific prompt update that names the top failure cluster it is meant to fix and could plausibly produce substantially better task performance than the parent.",
            "Shoot for large wins. Small prompt polish, extra canonical-output reminders, or mild clarifications are not acceptable except for the single conservative control.",
            "Generic output-contract reminders, canonical-label reminders, or paraphrases of the parent are wasted proposals unless paired with concrete task heuristics.",
            "Make at least half the proposals structurally different from the parent, not just longer."
        ],
        "instructions": format!("Create exactly proposals_per_round distinct candidates. Use frontier_variation for one Pareto-front parent and frontier_merge to combine two complementary Pareto-front parents from merge_candidate_pairs. If no merge pairs are available, replace requested merges with additional frontier variations. {} At most one proposal may be conservative; the rest must be very ambitious, task-specific changes aimed at named top failure clusters and designed to substantially outperform the parent. Make distinct candidates explore genuinely different strategies (structural rewrites, boundary taxonomies, conflict precedence, answer-routing procedures, few-shot examples when allowed, role priming, etc.) rather than paraphrasing one another.", proposer_policy_text(input)),
    })
}

fn sorted_pareto_member_ids(
    input: &CodexProposerInput<'_>,
    front: &CodexParetoFront,
) -> Vec<String> {
    let mut members = front.members.iter().cloned().collect::<Vec<_>>();
    if members.is_empty() {
        members = input
            .candidates
            .iter()
            .map(|candidate| candidate.candidate_id.clone())
            .collect();
    }
    members.sort_by(|left, right| {
        let left_wins = front.win_counts.get(left).copied().unwrap_or(0);
        let right_wins = front.win_counts.get(right).copied().unwrap_or(0);
        right_wins.cmp(&left_wins).then_with(|| left.cmp(right))
    });
    members
}

fn merge_candidate_pairs(members: &[String]) -> Vec<Vec<String>> {
    let mut pairs = Vec::new();
    for (left_index, left) in members.iter().enumerate() {
        for right in members.iter().skip(left_index + 1) {
            pairs.push(vec![left.clone(), right.clone()]);
        }
    }
    pairs
}

fn common_ancestor_id(input: &CodexProposerInput<'_>, candidate_ids: &[String]) -> String {
    let Some(first) = candidate_ids.first() else {
        return String::new();
    };
    let chains = candidate_ids
        .iter()
        .map(|candidate_id| ancestor_chain(input, candidate_id))
        .collect::<Vec<_>>();
    for candidate_id in ancestor_chain(input, first) {
        if chains
            .iter()
            .all(|chain| chain.iter().any(|item| item == &candidate_id))
        {
            return candidate_id;
        }
    }
    first.clone()
}

fn ancestor_chain(input: &CodexProposerInput<'_>, candidate_id: &str) -> Vec<String> {
    let mut chain = Vec::new();
    let mut seen = BTreeSet::new();
    let mut current = candidate_id.to_string();
    while seen.insert(current.clone()) {
        chain.push(current.clone());
        let Some(parent_id) = input
            .candidates
            .iter()
            .find(|candidate| candidate.candidate_id == current)
            .and_then(|candidate| candidate.parent_id.clone())
        else {
            break;
        };
        current = parent_id;
    }
    chain
}

fn candidate_selector_read_model(input: &CodexProposerInput<'_>) -> Value {
    json!({
        "name": normalize_candidate_selector_name(&input.config.gepa.candidate_selector.name),
        "configured_name": input.config.gepa.candidate_selector.name,
        "epsilon": input.config.gepa.candidate_selector.epsilon,
        "k": input.config.gepa.candidate_selector.k,
        "frontier_type": normalize_frontier_type(&input.config.gepa.frontier_type),
        "selection_objective": configured_selection_objective(input),
    })
}

fn batch_sampler_read_model(input: &CodexProposerInput<'_>) -> Value {
    json!({
        "name": normalize_batch_sampler_name(&input.config.gepa.batch_sampler.name),
        "configured_name": input.config.gepa.batch_sampler.name,
        "epoch_width": input.config.gepa.batch_sampler.epoch_width,
        "field": input.config.gepa.batch_sampler.field,
        "minibatch_size": input.config.gepa.minibatch_size,
        "proposals_per_round": input.config.gepa.proposals_per_generation,
        "objective_keys": input.config.gepa.objective_keys,
        "objective_directions": input.config.gepa.objective_directions,
    })
}

fn acceptance_read_model(input: &CodexProposerInput<'_>) -> Value {
    json!({
        "acceptance_criterion": normalize_acceptance_criterion(&input.config.gepa.acceptance_criterion),
        "configured_acceptance_criterion": input.config.gepa.acceptance_criterion,
        "minibatch_accept_margin": input.config.gepa.minibatch_accept_margin,
        "objective_directions": input.config.gepa.objective_directions,
        "objective_acceptance": {
            "min_objective_delta": input.config.gepa.objective_acceptance.min_objective_delta.unwrap_or(0.05),
            "objective_regression_tolerance": input.config.gepa.objective_acceptance.objective_regression_tolerance.unwrap_or(0.10),
            "protected_objectives": input.config.gepa.objective_acceptance.protected_objectives,
        },
    })
}

fn normalize_acceptance_criterion(criterion: &str) -> String {
    match criterion
        .trim()
        .to_ascii_lowercase()
        .replace('-', "_")
        .as_str()
    {
        "improvement_or_equal" => "improvement_or_equal".to_string(),
        "primary_or_objective" => "primary_or_objective".to_string(),
        "any_objective_improved" => "any_objective_improved".to_string(),
        "protected_objective_guard" => "protected_objective_guard".to_string(),
        _ => "primary_improvement".to_string(),
    }
}

fn task_pools_read_model(input: &CodexProposerInput<'_>) -> Value {
    if input.task_pool_rows.is_null() {
        return json!({});
    }
    input.task_pool_rows.clone()
}

fn task_pool_counts(input: &CodexProposerInput<'_>) -> Value {
    let mut counts = Map::new();
    if let Some(pools) = input.task_pool_rows.as_object() {
        for (name, pool) in pools {
            if name == "schema_version" {
                continue;
            }
            let row_count = pool
                .get("row_count")
                .and_then(Value::as_u64)
                .or_else(|| {
                    pool.get("rows")
                        .and_then(Value::as_array)
                        .map(|rows| rows.len() as u64)
                })
                .unwrap_or(0);
            counts.insert(name.clone(), json!(row_count));
        }
    }
    Value::Object(counts)
}

fn reflection_examples_read_model(input: &CodexProposerInput<'_>) -> Value {
    input
        .task_pool_rows
        .get("reflection")
        .and_then(|pool| pool.get("rows"))
        .and_then(Value::as_array)
        .map(|rows| Value::Array(rows.iter().take(40).cloned().collect()))
        .unwrap_or_else(|| Value::Array(Vec::new()))
}

fn candidates_read_model(input: &CodexProposerInput<'_>) -> Value {
    let pareto_front = compute_pareto_front(input);
    Value::Array(
        input
            .candidates
            .iter()
            .map(|candidate| {
                json!({
                    "candidate_id": candidate.candidate_id,
                    "parent_id": candidate.parent_id,
                    "source": candidate.source,
                    "status": candidate.status,
                    "is_parent": candidate.candidate_id == input.parent.candidate_id,
                    "is_pareto_front": pareto_front.members.contains(&candidate.candidate_id),
                    "payload": candidate.payload,
                    "minibatch_reward": candidate.minibatch_reward,
                    "train_reward": candidate.train_reward,
                    "heldout_reward": candidate.heldout_reward,
                    "minibatch_rollout_count": candidate.minibatch_scores.len(),
                    "train_rollout_count": candidate.train_scores.len(),
                    "sensor_frame_count": candidate.sensor_frames.len(),
                    "acceptance_score": candidate.acceptance_score,
                    "acceptance_metadata": candidate.acceptance_metadata,
                })
            })
            .collect(),
    )
}

fn candidate_deltas_read_model(input: &CodexProposerInput<'_>) -> Value {
    Value::Array(
        input
            .candidates
            .iter()
            .map(|candidate| {
                let parent_payload = input
                    .candidates
                    .iter()
                    .find(|parent| {
                        Some(parent.candidate_id.as_str()) == candidate.parent_id.as_deref()
                    })
                    .map(|parent| &parent.payload)
                    .unwrap_or(&input.parent.payload);
                let mut changed_modules = Vec::new();
                let mut module_deltas = Map::new();
                for module_id in &input.config.candidate.target_modules {
                    let before = parent_payload.get(module_id).cloned().unwrap_or_default();
                    let after = candidate
                        .payload
                        .get(module_id)
                        .cloned()
                        .unwrap_or_default();
                    if before != after {
                        changed_modules.push(module_id.clone());
                        module_deltas.insert(
                            module_id.clone(),
                            json!({
                                "before": before,
                                "after": after,
                            }),
                        );
                    }
                }
                json!({
                    "candidate_id": candidate.candidate_id,
                    "parent_id": candidate.parent_id,
                    "changed_modules": changed_modules,
                    "module_deltas": module_deltas,
                })
            })
            .collect(),
    )
}

fn rollouts_read_model(input: &CodexProposerInput<'_>) -> Value {
    let mut rows = Vec::new();
    for candidate in input.candidates {
        for score in &candidate.minibatch_scores {
            rows.push(json!({
                "candidate_id": candidate.candidate_id,
                "evaluation_stage": "candidate_minibatch",
                "example_id": score.example_id,
                "task_id": score.task_id,
                "reward": score.reward,
            }));
        }
        for score in &candidate.train_scores {
            rows.push(json!({
                "candidate_id": candidate.candidate_id,
                "evaluation_stage": "candidate_full_train",
                "example_id": score.example_id,
                "task_id": score.task_id,
                "reward": score.reward,
            }));
        }
        for frame in &candidate.sensor_frames {
            let rollout_trace = frame.metadata.get("rollout_trace").unwrap_or(&Value::Null);
            let summary = json_path(rollout_trace, &["summary"])
                .cloned()
                .unwrap_or(Value::Null);
            let outcome = json_path(rollout_trace, &["outcome"])
                .cloned()
                .unwrap_or_else(|| {
                    json!({
                        "status": frame.status,
                        "success_status": frame.success_status,
                        "reward": frame.reward,
                    })
                });
            let example = json_path(rollout_trace, &["task_payload", "example"])
                .cloned()
                .unwrap_or_else(|| {
                    json!({
                        "example_id": frame.example_id,
                        "task_id": frame.task_id,
                        "split": frame.split,
                    })
                });
            let reward_details = json_path(&outcome, &["reward_info", "details"])
                .or_else(|| frame.metadata.get("reward_details"))
                .cloned()
                .unwrap_or_else(|| json!({}));
            let expected = string_path(&summary, &["expected"])
                .or_else(|| string_path(&reward_details, &["expected"]))
                .or_else(|| string_path(&example, &["label"]))
                .unwrap_or_default();
            let prediction = string_path(&summary, &["prediction"])
                .or_else(|| string_path(&reward_details, &["prediction"]))
                .unwrap_or_default();
            let text = string_path(&example, &["text"]).unwrap_or_default();
            let policy_model = string_path(&reward_details, &["policy_model"])
                .or_else(|| string_path(&frame.usage, &["model"]))
                .unwrap_or_default();
            let rationale_text = frame
                .objective_scores
                .iter()
                .filter_map(|score| score.rationale.as_deref())
                .find(|value| !value.trim().is_empty())
                .unwrap_or_default()
                .to_string();
            let artifact_refs = frame
                .artifact_refs
                .iter()
                .map(|artifact| serde_json::to_value(artifact).unwrap_or(Value::Null))
                .filter(|value| !value.is_null())
                .collect::<Vec<_>>();
            let trace_refs = frame
                .trace_digest
                .as_ref()
                .map(|digest| vec![format!("trace_sha256:{}", digest.sha256)])
                .unwrap_or_default();
            rows.push(json!({
                "candidate_id": frame.candidate_id,
                "evaluation_stage": frame.evaluation_stage,
                "example_id": frame.example_id,
                "task_id": frame.task_id,
                "split": frame.split,
                "reward": frame.reward,
                "status": frame.status,
                "success_status": frame.success_status,
                "failure": frame.failure,
                "summary": summary,
                "outcome": outcome,
                "rationale_text": rationale_text,
                "expected": expected,
                "prediction": prediction,
                "text": text,
                "policy_model": policy_model,
                "usage": frame.usage,
                "artifact_refs": artifact_refs,
                "trace_refs": trace_refs,
                "candidate_status": candidate.status,
                "actionable_side_info": frame.actionable_side_info,
            }));
        }
    }
    Value::Array(rows)
}

fn proposer_examples_read_model(input: &CodexProposerInput<'_>) -> Value {
    let pareto_front = compute_pareto_front(input);
    let mut rows = Vec::new();
    for candidate in input.candidates {
        for frame in &candidate.sensor_frames {
            rows.push(proposer_example_row(input, &pareto_front, candidate, frame));
        }
    }
    rows.sort_by(|left, right| {
        proposer_example_sort_key(left).cmp(&proposer_example_sort_key(right))
    });
    Value::Array(rows)
}

fn proposer_example_row(
    input: &CodexProposerInput<'_>,
    pareto_front: &CodexParetoFront,
    candidate: &CandidateRecord,
    frame: &synth_optimizer_platform::SensorFrame,
) -> Value {
    let rollout_trace = frame.metadata.get("rollout_trace").unwrap_or(&Value::Null);
    let summary = json_path(rollout_trace, &["summary"])
        .cloned()
        .unwrap_or(Value::Null);
    let outcome = json_path(rollout_trace, &["outcome"])
        .cloned()
        .unwrap_or_else(|| {
            json!({
                "status": frame.status,
                "success_status": frame.success_status,
                "reward": frame.reward,
            })
        });
    let example = json_path(rollout_trace, &["task_payload", "example"])
        .cloned()
        .unwrap_or_else(|| {
            json!({
                "example_id": frame.example_id,
                "task_id": frame.task_id,
                "split": frame.split,
            })
        });
    let reward_details = json_path(&outcome, &["reward_info", "details"])
        .or_else(|| frame.metadata.get("reward_details"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let expected = string_path(&summary, &["expected"])
        .or_else(|| string_path(&reward_details, &["expected"]))
        .or_else(|| string_path(&example, &["label"]))
        .unwrap_or_default();
    let prediction = string_path(&summary, &["prediction"])
        .or_else(|| string_path(&reward_details, &["prediction"]))
        .unwrap_or_default();
    let text = string_path(&example, &["text"]).unwrap_or_default();
    let policy_model = string_path(&reward_details, &["policy_model"])
        .or_else(|| string_path(&frame.usage, &["model"]))
        .unwrap_or_default();
    let objective_rationale = frame
        .objective_scores
        .iter()
        .filter_map(|score| score.rationale.as_deref())
        .find(|value| !value.trim().is_empty())
        .unwrap_or_default()
        .to_string();
    let target_payload = input
        .config
        .candidate
        .target_modules
        .iter()
        .filter_map(|module_id| {
            candidate
                .payload
                .get(module_id)
                .map(|value| (module_id.clone(), Value::String(value.clone())))
        })
        .collect::<Map<_, _>>();
    let artifact_refs = frame
        .artifact_refs
        .iter()
        .map(|artifact| serde_json::to_value(artifact).unwrap_or(Value::Null))
        .filter(|value| !value.is_null())
        .collect::<Vec<_>>();
    let trace_refs = frame
        .trace_digest
        .as_ref()
        .map(|digest| vec![format!("trace_sha256:{}", digest.sha256)])
        .unwrap_or_default();
    json!({
        "schema_version": "gepa_proposer_example.v1",
        "candidate_id": candidate.candidate_id,
        "parent_candidate_id": candidate.parent_id,
        "candidate_status": candidate.status,
        "is_parent": candidate.candidate_id == input.parent.candidate_id,
        "is_pareto_front": pareto_front.members.contains(&candidate.candidate_id),
        "evaluation_stage": frame.evaluation_stage,
        "example_id": frame.example_id,
        "task_id": frame.task_id,
        "split": frame.split,
        "reward": frame.reward,
        "status": frame.status,
        "success_status": frame.success_status,
        "expected": expected,
        "prediction": prediction,
        "text": text,
        "policy_model": policy_model,
        "objective_rationale": objective_rationale,
        "failure": frame.failure,
        "usage": frame.usage,
        "target_payload": Value::Object(target_payload),
        "artifact_refs": artifact_refs,
        "trace_refs": trace_refs,
    })
}

fn proposer_failure_summary_read_model(
    input: &CodexProposerInput<'_>,
    proposer_examples: &Value,
) -> Value {
    let rows = proposer_examples.as_array().cloned().unwrap_or_default();
    let abstract_training_targets = literal_training_target_policy(input) == "forbid";
    let mut losses = Vec::new();
    let mut wins = Vec::new();
    let mut loss_labels: BTreeMap<String, usize> = BTreeMap::new();
    let mut win_labels: BTreeMap<String, usize> = BTreeMap::new();
    let mut confusion_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut candidate_stats: BTreeMap<String, ProposerCandidateEvidenceStats> = BTreeMap::new();

    for row in &rows {
        let candidate_id = string_path(row, &["candidate_id"]).unwrap_or_default();
        let expected = string_path(row, &["expected"]).unwrap_or_else(|| "unknown".to_string());
        let prediction = string_path(row, &["prediction"]).unwrap_or_else(|| "unknown".to_string());
        let expected_key = if abstract_training_targets {
            answer_value_bucket(&expected)
        } else {
            expected.clone()
        };
        let reward = row.get("reward").and_then(Value::as_f64).unwrap_or(0.0);
        let stats = candidate_stats.entry(candidate_id).or_default();
        stats.rollouts += 1;
        stats.reward_sum += reward;
        if reward >= 1.0 {
            stats.wins += 1;
            *win_labels.entry(expected_key).or_default() += 1;
            wins.push(proposer_example_compact(row));
        } else {
            stats.losses += 1;
            *loss_labels.entry(expected_key).or_default() += 1;
            *confusion_counts
                .entry(if abstract_training_targets {
                    abstract_target_confusion_key(&expected, &prediction)
                } else {
                    format!("{expected} -> {prediction}")
                })
                .or_default() += 1;
            losses.push(proposer_example_compact(row));
        }
    }

    let total_loss_count = losses.len();
    let total_win_count = wins.len();

    losses.sort_by(|left, right| {
        proposer_summary_sort_key(left).cmp(&proposer_summary_sort_key(right))
    });
    wins.sort_by(|left, right| {
        proposer_summary_sort_key(left).cmp(&proposer_summary_sort_key(right))
    });
    losses.truncate(80);
    wins.truncate(80);

    let candidate_outcomes = candidate_stats
        .into_iter()
        .map(|(candidate_id, stats)| {
            json!({
                "candidate_id": candidate_id,
                "rollouts": stats.rollouts,
                "wins": stats.wins,
                "losses": stats.losses,
                "average_reward": if stats.rollouts == 0 {
                    0.0
                } else {
                    stats.reward_sum / stats.rollouts as f64
                },
            })
        })
        .collect::<Vec<_>>();

    json!({
        "schema_version": "gepa_proposer_failure_summary.v1",
        "instructions": if abstract_training_targets {
            "Use this file first. The container task policy forbids literal training-target mappings: label_confusions are bucketed by output shape. Do not convert exact expected outputs or predictions from rollout evidence into prompt mappings."
        } else {
            "Use this file first. It is a flat, jq-friendly view of rollout evidence with text, expected output, prediction, reward, and prompt payload for wins/losses."
        },
        "literal_example_policy": proposer_literal_policy_json(input),
        "parent_candidate_id": input.parent.candidate_id,
        "target_modules": input.config.candidate.target_modules,
        "loss_count": total_loss_count,
        "win_count": total_win_count,
        "losses": losses,
        "wins": wins,
        "loss_labels": top_count_pairs(loss_labels, 40),
        "win_labels": top_count_pairs(win_labels, 40),
        "label_confusions": top_count_pairs(confusion_counts, 40),
        "candidate_outcomes": candidate_outcomes,
    })
}

fn proposer_repair_hints_read_model(
    input: &CodexProposerInput<'_>,
    proposer_examples: &Value,
) -> Value {
    let mut parent_loss_examples = Vec::new();
    let mut guard_win_examples = Vec::new();
    let mut confusion_clusters: BTreeMap<String, ProposerConfusionCluster> = BTreeMap::new();
    let rows = proposer_examples.as_array().cloned().unwrap_or_default();
    let abstract_training_targets = literal_training_target_policy(input) == "forbid";
    for row in &rows {
        let is_parent = row
            .get("is_parent")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let is_frontier = row
            .get("is_pareto_front")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !is_parent && !is_frontier {
            continue;
        }
        let reward = row.get("reward").and_then(Value::as_f64).unwrap_or(0.0);
        if reward >= 1.0 {
            guard_win_examples.push(proposer_reflection_example(row));
        } else {
            let expected = string_path(row, &["expected"]).unwrap_or_else(|| "unknown".to_string());
            let prediction =
                string_path(row, &["prediction"]).unwrap_or_else(|| "unknown".to_string());
            let key = if abstract_training_targets {
                abstract_target_confusion_key(&expected, &prediction)
            } else {
                format!("{expected} -> {prediction}")
            };
            let cluster =
                confusion_clusters
                    .entry(key)
                    .or_insert_with(|| ProposerConfusionCluster {
                        expected_label: if abstract_training_targets {
                            answer_value_bucket(&expected)
                        } else {
                            expected
                        },
                        observed_prediction: if abstract_training_targets {
                            answer_value_bucket(&prediction)
                        } else {
                            prediction
                        },
                        count: 0,
                        example_ids: Vec::new(),
                        candidate_ids: BTreeSet::new(),
                    });
            cluster.count += 1;
            if let Some(example_id) = string_path(row, &["example_id"]) {
                if cluster.example_ids.len() < 8 {
                    cluster.example_ids.push(example_id);
                }
            }
            if let Some(candidate_id) = string_path(row, &["candidate_id"]) {
                cluster.candidate_ids.insert(candidate_id);
            }
            parent_loss_examples.push(proposer_reflection_example(row));
        }
    }
    parent_loss_examples.sort_by(|left, right| {
        proposer_summary_sort_key(left).cmp(&proposer_summary_sort_key(right))
    });
    guard_win_examples.sort_by(|left, right| {
        proposer_summary_sort_key(left).cmp(&proposer_summary_sort_key(right))
    });
    parent_loss_examples.truncate(40);
    guard_win_examples.truncate(24);
    let mut clusters = confusion_clusters.into_values().collect::<Vec<_>>();
    clusters.sort_by(|left, right| {
        right.count.cmp(&left.count).then_with(|| {
            left.expected_label
                .cmp(&right.expected_label)
                .then_with(|| left.observed_prediction.cmp(&right.observed_prediction))
        })
    });
    clusters.truncate(40);
    json!({
        "schema_version": "gepa_proposer_repair_hints.v1",
        "instructions": if abstract_training_targets {
            "Use this after proposer_failure_summary. The task policy forbids literal training-target mappings: use rollout evidence to infer general rules, but do not lift observed exact targets into prompt tables."
        } else {
            "Use this after proposer_failure_summary. It is a reflection guide that highlights common confusions and guard wins. Use concrete examples only when the task_info/program policy says they are valid for this output space."
        },
        "literal_example_policy": proposer_literal_policy_json(input),
        "parent_candidate_id": input.parent.candidate_id,
        "target_modules": input.config.candidate.target_modules,
        "label_confusion_clusters": clusters.into_iter().map(ProposerConfusionCluster::into_json).collect::<Vec<_>>(),
        "parent_loss_examples": parent_loss_examples,
        "guard_win_examples": guard_win_examples,
        "proposal_guidance": [
            "Preserve the program's hard output contract (e.g. exact label format), but feel free to change everything else.",
            if abstract_training_targets { "Turn repeated confusions into general task procedures. Do not add literal train target mappings." } else { "Turn repeated confusions into explicit fixes: rules, output-disambiguation tables, or input→output few-shot pairs — whichever is most direct and task-valid." },
            if abstract_training_targets { "If examples are useful, make them abstract or synthetic examples rather than copied train input-target pairs." } else { "You may quote, paraphrase, or summarize training inputs as illustrative examples when that is the clearest way to teach a distinction." },
            "At most one proposal may be conservative. The remaining proposals should be very ambitious, high-variance, task-specific changes that could plausibly produce substantially better task performance than the parent.",
            "Shoot for large wins; mild polish and safe clarifications are wasted unless used as the single conservative control.",
            "Do not spend proposals on output-contract polish unless the dominant failures are output-format failures. Target the top confusion clusters directly.",
            "Make proposals meaningfully distinct from each other and from the parent — paraphrases of the seed are wasted budget."
        ],
    })
}

struct ProposerConfusionCluster {
    expected_label: String,
    observed_prediction: String,
    count: usize,
    example_ids: Vec<String>,
    candidate_ids: BTreeSet<String>,
}

impl ProposerConfusionCluster {
    fn into_json(self) -> Value {
        json!({
            "expected_label": self.expected_label,
            "observed_prediction": self.observed_prediction,
            "count": self.count,
            "example_ids": self.example_ids,
            "candidate_ids": self.candidate_ids.into_iter().collect::<Vec<_>>(),
            "reflection_prompt": "Diagnose and fix the confusion. First infer the task/output space from task_info and traces. Use concrete tables only for tasks where literal mappings are valid; otherwise write general procedures instead of target memorization.",
        })
    }
}

fn proposer_reflection_example(row: &Value) -> Value {
    json!({
        "candidate_id": row.get("candidate_id").cloned().unwrap_or(Value::Null),
        "candidate_status": row.get("candidate_status").cloned().unwrap_or(Value::Null),
        "is_parent": row.get("is_parent").cloned().unwrap_or(Value::Bool(false)),
        "is_pareto_front": row.get("is_pareto_front").cloned().unwrap_or(Value::Bool(false)),
        "evaluation_stage": row.get("evaluation_stage").cloned().unwrap_or(Value::Null),
        "example_id": row.get("example_id").cloned().unwrap_or(Value::Null),
        "seed": row.get("seed").cloned().unwrap_or(Value::Null),
        "split": row.get("split").cloned().unwrap_or(Value::Null),
        "reward": row.get("reward").cloned().unwrap_or(Value::Null),
        "expected": row.get("expected").cloned().unwrap_or(Value::Null),
        "prediction": row.get("prediction").cloned().unwrap_or(Value::Null),
        "policy_model": row.get("policy_model").cloned().unwrap_or(Value::Null),
        "objective_rationale": row.get("objective_rationale").cloned().unwrap_or(Value::Null),
        "artifact_refs": row.get("artifact_refs").cloned().unwrap_or(Value::Array(Vec::new())),
        "trace_refs": row.get("trace_refs").cloned().unwrap_or(Value::Array(Vec::new())),
        "text_policy": "Full text is in proposer_examples.json. Follow state/proposal_request.json.literal_example_policy before quoting or mapping examples inside candidate prompts.",
    })
}

fn proposer_metadata_read_model(
    input: &CodexProposerInput<'_>,
    rollouts: &Value,
    proposer_examples: &Value,
    proposer_failure_summary: &Value,
) -> Value {
    let pareto_front = compute_pareto_front(input);
    let proposer_model = input
        .config
        .proposer
        .model
        .clone()
        .unwrap_or_else(|| "gpt-5.4-mini".to_string());
    let workspace_root = input
        .workspace_dir
        .parent()
        .and_then(Path::parent)
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| input.workspace_dir.display().to_string());
    json!({
        "schema_version": "gepa_proposer_metadata.v1",
        "run_id": input.config.run.run_id,
        "generation": input.generation,
        "workspace_dir": input.workspace_dir,
        "workspace_root": workspace_root.clone(),
        "run_artifact_dir": workspace_root,
        "parent_candidate_id": input.parent.candidate_id,
        "frontier_size": pareto_front.members.len(),
        "frontier_type": pareto_front.frontier_type,
        "candidate_count": input.candidates.len(),
        "rollout_row_count": count_json_array(rollouts),
        "proposer_example_count": count_json_array(proposer_examples),
        "loss_count": proposer_failure_summary
            .get("loss_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        "win_count": proposer_failure_summary
            .get("win_count")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        "top_failures": proposer_failure_summary
            .get("label_confusions")
            .and_then(Value::as_array)
            .map(|items| Value::Array(items.iter().take(5).cloned().collect()))
            .unwrap_or_else(|| Value::Array(Vec::new())),
        "task_info": task_info_value(input).cloned().unwrap_or(Value::Null),
        "task_output_space_kind": task_output_space_kind(input),
        "literal_training_target_policy": literal_training_target_policy(input),
        "literal_example_policy": proposer_literal_policy_json(input),
        "target_modules": input.config.candidate.target_modules,
        "mutable_levers": input.program.mutable_field_ids(),
        "proposal_count": input.config.gepa.proposals_per_generation,
        "proposals_per_generation": input.config.gepa.proposals_per_generation,
        "minibatch_size": input.config.gepa.minibatch_size,
        "policy": {
            "provider": input.config.policy.provider,
            "model": input.config.policy.model,
            "api_family": input.config.policy.api_family,
            "base_url": input.config.policy.base_url,
            "inference_url": input.config.policy.inference_url,
            "max_tokens": input.config.policy.max_tokens,
            "disable_reasoning": input.config.policy.disable_reasoning,
            "tool_call_style": input.config.policy.tool_call_style,
            "proxy_mode": input.config.policy.proxy_mode,
            "credential_mode": input.config.policy.credential_mode,
            "config": input.config.policy.config,
        },
        "proposer": {
            "backend": input.config.proposer.backend,
            "model": proposer_model,
            "timeout_seconds": input.config.proposer.timeout_seconds,
            "sandbox_mode": input.config.proposer.sandbox_mode,
            "approval_policy": input.config.proposer.approval_policy,
        },
        "budgets": {
            "max_total_rollouts": input.config.gepa.effective_max_total_rollouts(),
            "max_train_rollouts": input.config.gepa.train_rollout_limit(),
            "max_heldout_rollouts": input.config.gepa.heldout_rollout_limit(),
            "max_cost_usd": input.config.gepa.max_cost_usd,
        },
        "task_pool_counts": task_pool_counts(input),
        "read_first": [
            "state/proposer_metadata.json",
            "state/task_info.json",
            "state/proposer_failure_summary.json",
            "state/proposer_repair_hints.json",
            "state/proposer_examples.json",
            "state/scores.json",
            "state/parent_payload.json",
            "state/candidate_deltas.json",
            "state/rollouts.json"
        ],
    })
}

fn proposer_readme_read_model() -> Value {
    json!({
        "schema_version": "gepa_proposer_readme.v1",
        "purpose": "Machine-readable index for the GEPA proposer workspace.",
        "read_order": [
            {
                "path": "prompting_best_practices.md",
                "use": "Shared GEPA proposer/reflector guidance: instruction typology, evidence-first loop, diagnosis rules, good changes, and bad changes."
            },
            {
                "path": "state/proposer_metadata.json",
                "use": "Small run/generation context, policy/proposer model names, counts, target modules, and top failure labels."
            },
            {
                "path": "state/task_info.json",
                "use": "Container-declared task description, output space, metrics, and proposer hints. Read this before deciding whether literal examples, tables, or abstract rules are valid."
            },
            {
                "path": "state/proposer_failure_summary.json",
                "use": "Primary flat evidence file. Start here for losses, wins, label confusions, expected/predicted labels, text, rewards, and target payloads."
            },
            {
                "path": "state/proposer_repair_hints.json",
                "use": "Reflection hints derived from parent/Pareto losses and guard wins. Use it to pick which confusions to fix. Follow state/proposal_request.json.literal_example_policy before quoting or mapping examples inside candidate prompts."
            },
            {
                "path": "state/proposer_examples.json",
                "use": "All flat rollout evidence rows with text, expected, prediction, reward, prompt payload, trace refs, and usage."
            },
            {
                "path": "state/scores.json",
                "use": "Candidate-level scores and rollout counts."
            },
            {
                "path": "state/parent_payload.json",
                "use": "The parent prompt payload to mutate."
            },
            {
                "path": "state/candidate_deltas.json",
                "use": "Prompt diffs between candidates and parents."
            },
            {
                "path": "state/rollouts.json",
                "use": "Per-rollout rows. Sensor-backed rows include summaries, outcomes, expected, prediction, text, rationale, and trace refs."
            },
            {
                "path": "state/reflective_frames.json",
                "use": "Nested reflective evidence under .frames[] for deeper trace-level detail."
            }
        ],
        "manifest_evidence_contract": [
            "List the files actually reviewed.",
            "Summarize parent/Pareto/recent candidate comparison.",
            "Name failure patterns grounded in losing examples.",
            "Name winning patterns grounded in successful examples.",
            "Cite concrete example_id values used."
        ],
        "candidate_prompt_contract": [
            "Read prompting_best_practices.md and classify each change as a premise, context, task_priority, core_task_description, heuristic, constraint, rule, input_description, or output_description.",
            "Ground the prompt in the rollout evidence; pull in any task-specific factual knowledge you can extract from wins and losses.",
            "Follow state/proposal_request.json.literal_example_policy. The valid use of concrete examples depends on the container-declared task/output space and the rollout traces.",
            "Mix strategies across proposals: structural rewrites, examples where task-valid, role priming, output-disambiguation rules, terse contracts — don't paraphrase the seed.",
            "Preserve any hard output contract (e.g. exact label format) declared in the program contract."
        ]
    })
}

fn proposer_example_compact(row: &Value) -> Value {
    json!({
        "candidate_id": row.get("candidate_id").cloned().unwrap_or(Value::Null),
        "candidate_status": row.get("candidate_status").cloned().unwrap_or(Value::Null),
        "is_parent": row.get("is_parent").cloned().unwrap_or(Value::Bool(false)),
        "is_pareto_front": row.get("is_pareto_front").cloned().unwrap_or(Value::Bool(false)),
        "evaluation_stage": row.get("evaluation_stage").cloned().unwrap_or(Value::Null),
        "example_id": row.get("example_id").cloned().unwrap_or(Value::Null),
        "seed": row.get("seed").cloned().unwrap_or(Value::Null),
        "split": row.get("split").cloned().unwrap_or(Value::Null),
        "reward": row.get("reward").cloned().unwrap_or(Value::Null),
        "expected": row.get("expected").cloned().unwrap_or(Value::Null),
        "prediction": row.get("prediction").cloned().unwrap_or(Value::Null),
        "text": row.get("text").cloned().unwrap_or(Value::Null),
        "policy_model": row.get("policy_model").cloned().unwrap_or(Value::Null),
        "target_payload": row.get("target_payload").cloned().unwrap_or(Value::Null),
        "artifact_refs": row.get("artifact_refs").cloned().unwrap_or(Value::Array(Vec::new())),
        "trace_refs": row.get("trace_refs").cloned().unwrap_or(Value::Array(Vec::new())),
    })
}

fn count_json_array(value: &Value) -> usize {
    value.as_array().map(Vec::len).unwrap_or(0)
}

fn proposer_example_sort_key(row: &Value) -> (String, String, i64, String) {
    (
        string_path(row, &["candidate_id"]).unwrap_or_default(),
        string_path(row, &["evaluation_stage"]).unwrap_or_default(),
        row.get("seed").and_then(Value::as_i64).unwrap_or(0),
        string_path(row, &["example_id"]).unwrap_or_default(),
    )
}

fn proposer_summary_sort_key(row: &Value) -> (String, i64, String) {
    (
        string_path(row, &["evaluation_stage"]).unwrap_or_default(),
        row.get("seed").and_then(Value::as_i64).unwrap_or(0),
        string_path(row, &["example_id"]).unwrap_or_default(),
    )
}

fn top_count_pairs(counts: BTreeMap<String, usize>, limit: usize) -> Value {
    let mut items = counts.into_iter().collect::<Vec<_>>();
    items.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    Value::Array(
        items
            .into_iter()
            .take(limit)
            .map(|(key, count)| json!({"key": key, "count": count}))
            .collect(),
    )
}

fn json_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for segment in path {
        current = current.get(*segment)?;
    }
    Some(current)
}

fn string_path(value: &Value, path: &[&str]) -> Option<String> {
    json_path(value, path)
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn task_info_value<'a>(input: &'a CodexProposerInput<'_>) -> Option<&'a Value> {
    input.program.metadata.get("task_info")
}

fn metadata_string_at(input: &CodexProposerInput<'_>, paths: &[&[&str]]) -> Option<String> {
    let metadata = Value::Object(input.program.metadata.clone());
    for path in paths {
        if let Some(value) = json_path(&metadata, path).and_then(Value::as_str) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn task_output_space_kind(input: &CodexProposerInput<'_>) -> Option<String> {
    metadata_string_at(
        input,
        &[
            &["task_info", "output_space", "kind"],
            &["task_info", "task", "output_space", "kind"],
            &["task_info", "metadata", "output_space", "kind"],
            &["task_info", "metadata", "task_output_space"],
            &["proposer_hints", "task_output_space"],
            &["proposer_constraints", "task_output_space"],
        ],
    )
}

fn literal_training_target_policy(input: &CodexProposerInput<'_>) -> &'static str {
    if let Some(policy) = metadata_string_at(
        input,
        &[
            &["task_info", "proposer_hints", "literal_training_targets"],
            &[
                "task_info",
                "metadata",
                "proposer_hints",
                "literal_training_targets",
            ],
            &["proposer_hints", "literal_training_targets"],
            &["proposer_constraints", "literal_training_targets"],
        ],
    ) {
        let policy = policy.to_ascii_lowercase();
        if policy.contains("forbid") || policy.contains("disallow") || policy.contains("avoid") {
            return "forbid";
        }
        if policy.contains("allow") {
            return "allow";
        }
    }
    if let Some(kind) = task_output_space_kind(input) {
        let kind = kind.to_ascii_lowercase();
        if kind.contains("finite")
            || kind.contains("label")
            || kind.contains("closed")
            || kind.contains("intent")
            || kind.contains("class")
        {
            return "allow";
        }
        if kind.contains("open")
            || kind.contains("free")
            || kind.contains("answer")
            || kind.contains("extract")
            || kind.contains("span")
            || kind.contains("generat")
        {
            return "forbid";
        }
    }
    if input.program.metadata.get("labels").is_some()
        || input.program.metadata.get("label_guidance").is_some()
    {
        return "allow";
    }
    let primary_metric = metadata_string_at(
        input,
        &[
            &["task_info", "metadata", "primary_metric"],
            &["primary_metric"],
        ],
    )
    .unwrap_or_default()
    .to_ascii_lowercase();
    let answer_contract = metadata_string_at(
        input,
        &[
            &["task_info", "metadata", "answer_contract"],
            &["answer_contract"],
        ],
    )
    .unwrap_or_default()
    .to_ascii_lowercase();
    if primary_metric.contains("f1")
        || primary_metric.contains("rouge")
        || answer_contract.contains("answer")
        || answer_contract.contains("span")
    {
        return "forbid";
    }
    "infer"
}

fn proposer_literal_policy_json(input: &CodexProposerInput<'_>) -> Value {
    json!({
        "source": "container /task_info, /program metadata, and rollout trace evidence",
        "task_output_space_kind": task_output_space_kind(input),
        "literal_training_targets": literal_training_target_policy(input),
        "policy": [
            "First infer the task and output space from state/program_contract.json.metadata.task_info, state/proposer_metadata.json, rollout traces, expected outputs, predictions, and objective rationales.",
            "If the task has a finite closed output space, concrete boundary examples or compact output mappings can be valid when they teach the output contract.",
            "If the task is not a finite closed-output mapping, convert trace evidence into general procedures and do not copy observed training targets into the prompt.",
            "When task_info.proposer_hints is present, treat it as authoritative unless trace evidence contradicts it."
        ],
    })
}

fn proposer_policy_text(input: &CodexProposerInput<'_>) -> String {
    let output_kind = task_output_space_kind(input).unwrap_or_else(|| "unspecified".to_string());
    let literal_policy = literal_training_target_policy(input);
    format!(
        "Task policy: infer the task from container task_info, program metadata, rollout traces, expected outputs, predictions, and objective rationales before proposing. Output-space kind from the container is {output_kind:?}; literal-training-target policy is {literal_policy:?}. Finite closed-output tasks may use concrete boundary examples or output tables when they generalize. Non-closed-output tasks should turn traces into general procedures and avoid copying observed training targets into candidate prompts. Follow any task_info.proposer_hints fields."
    )
}

fn answer_value_bucket(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return "empty answer".to_string();
    }
    if trimmed.eq_ignore_ascii_case("yes") || trimmed.eq_ignore_ascii_case("no") {
        return "yes/no answer".to_string();
    }
    if trimmed.chars().any(|ch| ch.is_ascii_digit()) {
        return "date/number answer".to_string();
    }
    let word_count = trimmed.split_whitespace().count();
    if word_count >= 8 {
        return "long clause answer".to_string();
    }
    if trimmed
        .chars()
        .next()
        .map(|ch| ch.is_uppercase())
        .unwrap_or(false)
    {
        return "proper-name/title answer".to_string();
    }
    if word_count <= 3 {
        "short phrase/category answer".to_string()
    } else {
        "clause answer".to_string()
    }
}

fn abstract_target_confusion_key(expected: &str, prediction: &str) -> String {
    format!(
        "{} expected -> {} prediction",
        answer_value_bucket(expected),
        answer_value_bucket(prediction)
    )
}

fn literal_target_is_specific(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.len() < 4 || trimmed.len() > 96 {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    !matches!(
        lower.as_str(),
        "yes" | "no" | "unknown" | "none" | "true" | "false" | "n/a"
    )
}

fn scores_read_model(input: &CodexProposerInput<'_>) -> Value {
    Value::Array(
        input
            .candidates
            .iter()
            .map(|candidate| {
                json!({
                    "candidate_id": candidate.candidate_id,
                    "status": candidate.status,
                    "source": candidate.source,
                    "minibatch_reward": candidate.minibatch_reward,
                    "train_reward": candidate.train_reward,
                    "heldout_reward": candidate.heldout_reward,
                    "rollout_counts": {
                        "minibatch": candidate.minibatch_scores.len(),
                        "train": candidate.train_scores.len(),
                        "sensor_frames": candidate.sensor_frames.len(),
                    },
                })
            })
            .collect(),
    )
}

fn evidence_frames_read_model(input: &CodexProposerInput<'_>) -> Value {
    Value::Array(
        input
            .candidates
            .iter()
            .flat_map(|candidate| candidate.sensor_frames.iter())
            .map(|frame| serde_json::to_value(frame).unwrap_or(Value::Null))
            .filter(|value| !value.is_null())
            .collect(),
    )
}

fn reflective_frames_read_model(input: &CodexProposerInput<'_>) -> Value {
    let mut frames = input
        .candidates
        .iter()
        .flat_map(|candidate| {
            candidate
                .sensor_frames
                .iter()
                .map(move |frame| reflective_frame_value(input, candidate, frame))
        })
        .collect::<Vec<_>>();
    frames.sort_by(|left, right| {
        let left_key = left
            .get("frame_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let right_key = right
            .get("frame_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        left_key.cmp(right_key)
    });
    if frames.len() > 80 {
        frames.truncate(80);
    }
    json!({
        "schema_version": GEPA_REFLECTIVE_FRAME_SCHEMA_VERSION,
        "adapter": reflective_adapter_spec(),
        "frame_count": frames.len(),
        "frames": frames,
    })
}

fn reflective_frame_value(
    input: &CodexProposerInput<'_>,
    candidate: &CandidateRecord,
    frame: &synth_optimizer_platform::SensorFrame,
) -> Value {
    let component_id = input
        .config
        .candidate
        .target_modules
        .first()
        .cloned()
        .unwrap_or_else(|| "candidate".to_string());
    let rollout_id = frame.rollout_id.clone().unwrap_or_default();
    let trace_refs = frame
        .trace_digest
        .as_ref()
        .map(|digest| vec![format!("trace_sha256:{}", digest.sha256)])
        .unwrap_or_default();
    let confidence = reflective_confidence(frame);
    let artifact_refs = frame
        .artifact_refs
        .iter()
        .map(|artifact| serde_json::to_value(artifact).unwrap_or(Value::Null))
        .filter(|value| !value.is_null())
        .collect::<Vec<_>>();
    let failure_class = frame
        .failure
        .as_ref()
        .map(|failure| failure.failure_class().to_string())
        .unwrap_or_default();
    let verifier_rationale = frame
        .objective_scores
        .iter()
        .filter_map(|score| score.rationale.as_deref())
        .find(|value| !value.trim().is_empty())
        .unwrap_or_default()
        .to_string();
    let rollout_trace = frame
        .metadata
        .get("rollout_trace")
        .and_then(Value::as_object);
    let trace_summary = rollout_trace
        .and_then(|trace| trace.get("summary"))
        .cloned()
        .or_else(|| frame.metadata.get("summary").cloned())
        .unwrap_or(Value::Null);
    let trace_outcome = rollout_trace
        .and_then(|trace| trace.get("outcome"))
        .cloned()
        .unwrap_or_else(|| {
            json!({
                "status": frame.status,
                "success_status": frame.success_status,
                "reward": frame.reward,
            })
        });
    let task_example = rollout_trace
        .and_then(|trace| trace.get("task_payload"))
        .and_then(|task_payload| task_payload.get("example"))
        .cloned()
        .unwrap_or_else(|| {
            json!({
                "example_id": frame.example_id,
                "task_id": frame.task_id,
                "split": frame.split,
            })
        });
    let request = rollout_trace
        .and_then(|trace| trace.get("request"))
        .cloned()
        .unwrap_or_else(|| {
            json!({
                "evaluation_stage": frame.evaluation_stage,
                "target_modules": input.config.candidate.target_modules,
            })
        });
    let tool_calls = rollout_trace
        .and_then(|trace| trace.get("tool_calls"))
        .cloned()
        .unwrap_or_else(|| json!([]));
    let substitution_stats = rollout_trace
        .and_then(|trace| trace.get("substitution_stats"))
        .cloned()
        .unwrap_or_else(|| json!({"attempted": 0, "applied": 0, "warnings": []}));
    let evidence = json!({
        "schema_version": GEPA_REFLECTIVE_FRAME_SCHEMA_VERSION,
        "source": "sensor_frame_adapter",
        "adapter": reflective_adapter_spec(),
        "subject": {
            "algorithm_id": GEPA_ALGORITHM_ID,
            "candidate_id": candidate.candidate_id,
            "parent_candidate_id": candidate.parent_id,
            "component_id": component_id,
            "rollout_id": rollout_id,
            "example_id": frame.example_id,
        },
        "adapter_source": GEPA_ADAPTER_SOURCE,
        "rollout_id": rollout_id,
        "example_id": frame.example_id,
        "split": frame.split,
        "inputs": {
            "example": task_example,
            "request": request,
        },
        "generated_outputs": {
            "summary": trace_summary,
            "outcome": trace_outcome,
        },
        "feedback": {
            "reward": frame.reward,
            "objective_scores": frame.objective_scores,
            "verifier_rationale": verifier_rationale,
        },
        "actionable_side_info": frame.actionable_side_info.clone().unwrap_or_else(|| json!({})),
        "sensors": {
            "confidence": confidence,
            "trace_digest": frame.trace_digest,
        },
        "refs": {
            "trace_refs": trace_refs,
            "rollout_id": rollout_id,
            "sensor_frame_id": frame.sensor_frame_id,
            "artifact_refs": artifact_refs,
        },
        "trace_refs": trace_refs,
        "tool_calls": tool_calls,
        "substitution_stats": substitution_stats,
        "failure_class": failure_class,
        "usage": frame.usage,
        "confidence": confidence,
        "component_id": component_id,
    });
    json!({
        "frame_id": format!("reflect:{}:{}:{}", GEPA_ALGORITHM_ID, candidate.candidate_id, frame.sensor_frame_id),
        "algorithm_id": GEPA_ALGORITHM_ID,
        "component_id": component_id,
        "candidate_id": candidate.candidate_id,
        "parent_candidate_id": candidate.parent_id,
        "rollout_id": frame.rollout_id,
        "artifact_refs": artifact_refs,
        "metadata": {
            "adapter_id": CONTAINER_SENSOR_ADAPTER_ID,
            "adapter_version": CONTAINER_SENSOR_ADAPTER_VERSION,
            "evidence_schema_version": GEPA_REFLECTIVE_FRAME_SCHEMA_VERSION,
            "sensor_frame_id": frame.sensor_frame_id,
        },
        "evidence": evidence,
    })
}

fn reflective_adapter_spec() -> Value {
    json!({
        "adapter_id": CONTAINER_SENSOR_ADAPTER_ID,
        "adapter_version": CONTAINER_SENSOR_ADAPTER_VERSION,
        "source": GEPA_ADAPTER_SOURCE,
        "evidence_schema_version": GEPA_REFLECTIVE_FRAME_SCHEMA_VERSION,
        "required_evidence_keys": [
            "schema_version",
            "source",
            "adapter",
            "subject",
            "inputs",
            "generated_outputs",
            "feedback",
            "actionable_side_info",
            "sensors",
            "refs",
        ],
    })
}

fn reflective_confidence(frame: &synth_optimizer_platform::SensorFrame) -> f64 {
    let support_count = frame
        .trace_digest
        .as_ref()
        .map(|digest| digest.llm_request_count + digest.tool_call_count)
        .unwrap_or(0);
    if frame.failure.is_some() {
        0.55
    } else if support_count > 0 {
        0.85
    } else if frame.actionable_side_info.is_some() {
        0.7
    } else {
        0.35
    }
}

fn links_read_model(input: &CodexProposerInput<'_>) -> Value {
    let mut links = Vec::new();
    for candidate in input.candidates {
        if let Some(parent_id) = &candidate.parent_id {
            links.push(json!({
                "type": "candidate_parent",
                "from": candidate.candidate_id,
                "to": parent_id,
            }));
        }
        for frame in &candidate.sensor_frames {
            links.push(json!({
                "type": "candidate_rollout_evidence",
                "from": candidate.candidate_id,
                "to": frame.sensor_frame_id,
                "example_id": frame.example_id,
                "evaluation_stage": frame.evaluation_stage,
            }));
        }
    }
    Value::Array(links)
}

fn pareto_front_read_model(input: &CodexProposerInput<'_>) -> Value {
    let pareto_front = compute_pareto_front(input);
    let mut members = pareto_front
        .members
        .iter()
        .filter_map(|candidate_id| {
            input
                .candidates
                .iter()
                .find(|candidate| &candidate.candidate_id == candidate_id)
                .map(|candidate| {
                    let win_count = pareto_front
                        .win_counts
                        .get(&candidate.candidate_id)
                        .copied()
                        .unwrap_or(0);
                    json!({
                        "candidate_id": candidate.candidate_id,
                        "parent_id": candidate.parent_id,
                        "source": candidate.source,
                        "status": candidate.status,
                        "train_reward": candidate.train_reward,
                        "minibatch_reward": candidate.minibatch_reward,
                        "heldout_reward": candidate.heldout_reward,
                        "win_count": win_count,
                        "payload": candidate.payload,
                    })
                })
        })
        .collect::<Vec<_>>();
    members.sort_by(|left, right| {
        left.get("candidate_id")
            .and_then(Value::as_str)
            .cmp(&right.get("candidate_id").and_then(Value::as_str))
    });
    json!({
        "schema_version": "gepa_pareto_front.v1",
        "frontier_type": pareto_front.frontier_type,
        "score_source": pareto_front.score_source,
        "objective_keys": input.config.gepa.objective_keys,
        "objective_directions": input.config.gepa.objective_directions,
        "parent_candidate_id": input.parent.candidate_id,
        "candidate_selector": candidate_selector_read_model(input),
        "members": members,
        "win_counts": pareto_front.win_counts,
        "cells": pareto_front.cells,
        "legacy_status_frontier": legacy_frontier_read_model(input),
    })
}

#[derive(Debug)]
struct CodexParetoFront {
    frontier_type: String,
    score_source: String,
    members: BTreeSet<String>,
    win_counts: BTreeMap<String, usize>,
    cells: Vec<Value>,
}

fn compute_pareto_front(input: &CodexProposerInput<'_>) -> CodexParetoFront {
    let frontier_type = normalize_frontier_type(&input.config.gepa.frontier_type);
    let mut cells = match frontier_type.as_str() {
        "per_objective" => codex_pareto_objective_cells(input),
        "per_example_objective" => codex_pareto_example_objective_cells(input),
        _ => codex_pareto_example_cells(input),
    };
    if cells.is_empty() && frontier_type != "per_example" {
        cells = codex_pareto_example_cells(input);
    }
    let mut members = BTreeSet::new();
    let mut win_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut cell_values = Vec::new();
    for cell in cells {
        members.insert(cell.candidate_id.clone());
        *win_counts.entry(cell.candidate_id.clone()).or_default() += 1;
        cell_values.push(json!({
            "frontier_key": cell.frontier_key,
            "candidate_id": cell.candidate_id,
            "score": cell.score,
            "example_id": cell.example_id,
            "objective_id": cell.objective_id,
        }));
    }
    if members.is_empty() {
        for candidate in input.candidates {
            if candidate.train_reward.is_some() {
                members.insert(candidate.candidate_id.clone());
                win_counts.insert(candidate.candidate_id.clone(), 1);
                cell_values.push(json!({
                    "frontier_key": format!("candidate:{}", candidate.candidate_id),
                    "candidate_id": candidate.candidate_id,
                    "score": candidate.train_reward,
                    "example_id": Value::Null,
                    "objective_id": Value::Null,
                }));
            }
        }
    }
    CodexParetoFront {
        frontier_type,
        score_source: "sensor_frame.objective_scores".to_string(),
        members,
        win_counts,
        cells: cell_values,
    }
}

#[derive(Clone, Debug)]
struct CodexParetoCell {
    frontier_key: String,
    candidate_id: String,
    score: f64,
    example_id: Option<String>,
    objective_id: Option<String>,
}

fn codex_pareto_example_cells(input: &CodexProposerInput<'_>) -> Vec<CodexParetoCell> {
    let selection_objective = configured_selection_objective(input);
    let selection_direction = selection_objective
        .as_deref()
        .map(|objective| codex_objective_direction(input, objective))
        .unwrap_or(1.0);
    let mut winners: BTreeMap<String, CodexParetoCell> = BTreeMap::new();
    for candidate in input.candidates {
        if candidate.train_reward.is_none() {
            continue;
        }
        for frame in train_sensor_frames(candidate) {
            let candidate_id = candidate.candidate_id.clone();
            let score = frame_objective_score(frame, selection_objective.as_deref())
                .unwrap_or(frame.reward);
            upsert_codex_pareto_cell(
                &mut winners,
                frame.example_id.clone(),
                CodexParetoCell {
                    frontier_key: format!("example:{}", frame.example_id),
                    candidate_id,
                    score: score * selection_direction,
                    example_id: Some(frame.example_id.clone()),
                    objective_id: None,
                },
            );
        }
    }
    winners.into_values().collect()
}

fn codex_pareto_objective_cells(input: &CodexProposerInput<'_>) -> Vec<CodexParetoCell> {
    let objective_keys = configured_objective_keys(input);
    let mut sums: BTreeMap<(String, String), (f64, usize)> = BTreeMap::new();
    for candidate in input.candidates {
        if candidate.train_reward.is_none() {
            continue;
        }
        for frame in train_sensor_frames(candidate) {
            for score in &frame.objective_scores {
                if !objective_keys.is_empty() && !objective_keys.contains(&score.objective) {
                    continue;
                }
                let entry = sums
                    .entry((candidate.candidate_id.clone(), score.objective.clone()))
                    .or_insert((0.0, 0));
                entry.0 += score.value;
                entry.1 += 1;
            }
        }
    }
    let mut winners = BTreeMap::new();
    for ((candidate_id, objective), (sum, count)) in sums {
        if count == 0 {
            continue;
        }
        upsert_codex_pareto_cell(
            &mut winners,
            objective.clone(),
            CodexParetoCell {
                frontier_key: format!("objective:{objective}"),
                candidate_id,
                score: (sum / count as f64) * codex_objective_direction(input, &objective),
                example_id: None,
                objective_id: Some(objective),
            },
        );
    }
    winners.into_values().collect()
}

fn codex_pareto_example_objective_cells(input: &CodexProposerInput<'_>) -> Vec<CodexParetoCell> {
    let objective_keys = configured_objective_keys(input);
    let mut winners = BTreeMap::new();
    for candidate in input.candidates {
        if candidate.train_reward.is_none() {
            continue;
        }
        for frame in train_sensor_frames(candidate) {
            for score in &frame.objective_scores {
                if !objective_keys.is_empty() && !objective_keys.contains(&score.objective) {
                    continue;
                }
                let key = format!("{}|{}", frame.example_id, score.objective);
                upsert_codex_pareto_cell(
                    &mut winners,
                    key,
                    CodexParetoCell {
                        frontier_key: format!(
                            "example_objective:{}|{}",
                            frame.example_id, score.objective
                        ),
                        candidate_id: candidate.candidate_id.clone(),
                        score: score.value * codex_objective_direction(input, &score.objective),
                        example_id: Some(frame.example_id.clone()),
                        objective_id: Some(score.objective.clone()),
                    },
                );
            }
        }
    }
    winners.into_values().collect()
}

fn train_sensor_frames(
    candidate: &CandidateRecord,
) -> impl Iterator<Item = &synth_optimizer_platform::SensorFrame> {
    candidate.sensor_frames.iter().filter(|frame| {
        matches!(
            frame.evaluation_stage.as_str(),
            "seed_full_train" | "candidate_full_train"
        )
    })
}

fn upsert_codex_pareto_cell(
    winners: &mut BTreeMap<String, CodexParetoCell>,
    key: String,
    challenger: CodexParetoCell,
) {
    let should_replace = winners
        .get(&key)
        .map(|incumbent| {
            challenger.score > incumbent.score + f64::EPSILON
                || ((challenger.score - incumbent.score).abs() <= f64::EPSILON
                    && challenger.candidate_id < incumbent.candidate_id)
        })
        .unwrap_or(true);
    if should_replace {
        winners.insert(key, challenger);
    }
}

fn configured_selection_objective(input: &CodexProposerInput<'_>) -> Option<String> {
    input
        .config
        .gepa
        .selection_objective
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn configured_objective_keys(input: &CodexProposerInput<'_>) -> BTreeSet<String> {
    input
        .config
        .gepa
        .objective_keys
        .iter()
        .map(|objective| objective.trim())
        .filter(|objective| !objective.is_empty())
        .map(str::to_string)
        .collect()
}

fn codex_objective_direction(input: &CodexProposerInput<'_>, objective: &str) -> f64 {
    input
        .config
        .gepa
        .objective_directions
        .get(objective)
        .map(String::as_str)
        .map(normalize_objective_direction)
        .unwrap_or(1.0)
}

fn normalize_objective_direction(direction: &str) -> f64 {
    match direction.trim().to_ascii_lowercase().as_str() {
        "min" | "minimize" | "lower" | "lower_is_better" | "down" => -1.0,
        _ => 1.0,
    }
}

fn frame_objective_score(
    frame: &synth_optimizer_platform::SensorFrame,
    objective: Option<&str>,
) -> Option<f64> {
    let objective = objective?;
    frame
        .objective_scores
        .iter()
        .find(|score| score.objective == objective)
        .map(|score| score.value)
}

fn normalize_frontier_type(value: &str) -> String {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "per_objective" => "per_objective".to_string(),
        "per_example_objective" => "per_example_objective".to_string(),
        _ => "per_example".to_string(),
    }
}

fn normalize_candidate_selector_name(value: &str) -> String {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "pareto" | "pareto_weighted" => "pareto_weighted".to_string(),
        "uniform_pareto" => "uniform_pareto".to_string(),
        "random" => "random".to_string(),
        "current_best" => "current_best".to_string(),
        "top_k_pareto" => "top_k_pareto".to_string(),
        "epsilon_greedy" => "epsilon_greedy".to_string(),
        _ => "pareto_weighted".to_string(),
    }
}

fn normalize_batch_sampler_name(value: &str) -> String {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "epoch_shuffled" => "epoch_shuffled".to_string(),
        "ordered_epoch" | "sequential_epoch" => "ordered_epoch".to_string(),
        "stratified" | "stratified_by_field" => "stratified".to_string(),
        _ => "seeded_shuffle".to_string(),
    }
}

fn gepa_summary_read_model(input: &CodexProposerInput<'_>, rollouts: &Value) -> Value {
    let pareto_front = compute_pareto_front(input);
    let best = input.candidates.iter().max_by(|left, right| {
        score_for_order(left)
            .partial_cmp(&score_for_order(right))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let example_ids = rollouts
        .as_array()
        .map(|rows| {
            rows.iter()
                .filter_map(|row| row.get("example_id").and_then(Value::as_str))
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    json!({
        "candidate_count": input.candidates.len(),
        "frontier_count": pareto_front.members.len(),
        "frontier_type": pareto_front.frontier_type,
        "candidate_selector": candidate_selector_read_model(input),
        "batch_sampler": batch_sampler_read_model(input),
        "parent_candidate_id": input.parent.candidate_id,
        "best_candidate_id": best.map(|candidate| candidate.candidate_id.as_str()),
        "best_train_reward": best.and_then(|candidate| candidate.train_reward),
        "observed_example_count": example_ids.len(),
        "rollout_row_count": rollouts.as_array().map(Vec::len).unwrap_or(0),
    })
}

fn legacy_frontier_read_model(input: &CodexProposerInput<'_>) -> Value {
    Value::Array(
        input
            .candidates
            .iter()
            .filter(|candidate| {
                candidate.status == "accepted"
                    || candidate.status == "seed"
                    || candidate.heldout_reward.is_some()
            })
            .map(|candidate| {
                json!({
                    "candidate_id": candidate.candidate_id,
                    "train_reward": candidate.train_reward,
                    "minibatch_reward": candidate.minibatch_reward,
                    "heldout_reward": candidate.heldout_reward,
                    "payload": candidate.payload,
                })
            })
            .collect(),
    )
}

fn reflector_input_read_model(
    input: &CodexProposerInput<'_>,
    prompting_best_practices: &str,
) -> Value {
    let mut wins = Vec::new();
    let mut losses = Vec::new();
    let proposer_examples = proposer_examples_read_model(input);
    let proposal_policy = proposer_policy_text(input);
    for row in proposer_examples.as_array().into_iter().flatten() {
        let sample = proposer_example_compact(row);
        if row.get("reward").and_then(Value::as_f64).unwrap_or(0.0) >= 1.0 {
            wins.push(sample);
        } else {
            losses.push(sample);
        }
    }
    wins.truncate(20);
    losses.truncate(20);
    json!({
        "parent_candidate_id": input.parent.candidate_id,
        "target_modules": input.config.candidate.target_modules,
        "sample_winning_traces": wins,
        "sample_losing_traces": losses,
        "wins": wins,
        "losses": losses,
        "prompting_best_practices": prompting_best_practices,
        "instructions": [
            "Read prompting_best_practices.md before diagnosing prompt changes.",
            "Classify likely edits using the shared typology: premise, context, task_priority, core_task_description, heuristics, constraints, rules, input_description, output_description.",
            "Use sampled wins and losses to drive substantive prompt changes — not paraphrases of the seed.",
            "At most one proposal may be conservative. The others must be very ambitious task-specific changes that attack named top failure clusters and could substantially outperform the parent.",
            proposal_policy,
            "Across proposals, explore different strategies (structural rewrite, few-shot examples, role priming, label-disambiguation table, terse contract). Distinct proposals should be genuinely distinct."
        ],
    })
}

fn score_for_order(candidate: &CandidateRecord) -> f64 {
    candidate
        .train_reward
        .or(candidate.minibatch_reward)
        .or(candidate.heldout_reward)
        .unwrap_or(f64::NEG_INFINITY)
}

fn thread_start_params(input: &CodexProposerInput<'_>, model: &str) -> Value {
    let mut params = Map::new();
    params.insert("model".to_string(), Value::String(model.to_string()));
    params.insert(
        "instructions".to_string(),
        Value::String(
            "You are the GEPA workspace proposer. Work only inside this workspace.".to_string(),
        ),
    );
    if let Some(approval_policy) = non_empty(input.config.proposer.approval_policy.as_deref()) {
        params.insert(
            "approvalPolicy".to_string(),
            Value::String(approval_policy.to_string()),
        );
    }
    if let Some(sandbox_mode) = non_empty(input.config.proposer.sandbox_mode.as_deref()) {
        params.insert(
            "sandbox".to_string(),
            Value::String(sandbox_mode.to_string()),
        );
    }
    Value::Object(params)
}

fn staleness_thread_start_params(input: &CodexStalenessReviewerInput<'_>, model: &str) -> Value {
    let mut params = Map::new();
    params.insert("model".to_string(), Value::String(model.to_string()));
    params.insert(
        "instructions".to_string(),
        Value::String(
            "You are the GEPA FlashEvolve staleness reviewer. Work only inside this workspace."
                .to_string(),
        ),
    );
    if let Some(approval_policy) = non_empty(input.config.proposer.approval_policy.as_deref()) {
        params.insert(
            "approvalPolicy".to_string(),
            Value::String(approval_policy.to_string()),
        );
    }
    if let Some(sandbox_mode) = non_empty(input.config.proposer.sandbox_mode.as_deref()) {
        params.insert(
            "sandbox".to_string(),
            Value::String(sandbox_mode.to_string()),
        );
    }
    Value::Object(params)
}

fn turn_start_params(input: &CodexProposerInput<'_>, model: &str) -> Result<Value> {
    let mut params = Map::new();
    params.insert("model".to_string(), Value::String(model.to_string()));
    params.insert(
        "input".to_string(),
        Value::Array(vec![json!({
            "type": "text",
            "text": proposer_instructions(input)?,
            "textElements": [],
        })]),
    );
    if let Some(reasoning_effort) = non_empty(input.config.proposer.reasoning_effort.as_deref()) {
        params.insert(
            "effort".to_string(),
            Value::String(reasoning_effort.to_string()),
        );
    }
    if let Some(approval_policy) = non_empty(input.config.proposer.approval_policy.as_deref()) {
        params.insert(
            "approvalPolicy".to_string(),
            Value::String(approval_policy.to_string()),
        );
    }
    if let Some(sandbox_mode) = non_empty(input.config.proposer.sandbox_mode.as_deref()) {
        params.insert(
            "sandboxPolicy".to_string(),
            sandbox_policy_for_mode(sandbox_mode),
        );
    }
    Ok(Value::Object(params))
}

fn staleness_turn_start_params(input: &CodexStalenessReviewerInput<'_>, model: &str) -> Value {
    let mut params = Map::new();
    params.insert("model".to_string(), Value::String(model.to_string()));
    params.insert(
        "input".to_string(),
        Value::Array(vec![json!({
            "type": "text",
            "text": staleness_reviewer_instructions(input),
            "textElements": [],
        })]),
    );
    if let Some(reasoning_effort) = non_empty(input.config.proposer.reasoning_effort.as_deref()) {
        params.insert(
            "effort".to_string(),
            Value::String(reasoning_effort.to_string()),
        );
    }
    if let Some(approval_policy) = non_empty(input.config.proposer.approval_policy.as_deref()) {
        params.insert(
            "approvalPolicy".to_string(),
            Value::String(approval_policy.to_string()),
        );
    }
    if let Some(sandbox_mode) = non_empty(input.config.proposer.sandbox_mode.as_deref()) {
        params.insert(
            "sandboxPolicy".to_string(),
            sandbox_policy_for_mode(sandbox_mode),
        );
    }
    Value::Object(params)
}

fn proposer_instructions(input: &CodexProposerInput<'_>) -> Result<String> {
    let context = proposer_prompt_context(input);
    let proposal_policy = proposer_policy_text(input);
    let best_practices = resolved_prompting_best_practices(input)?;
    let best_practices = best_practices.trim();
    Ok(format!(
        "{context}\n\n\
         Prompting best practices:\n\
         {best_practices}\n\n\
         Read README.md, prompting_best_practices.md, proposal/PROPOSAL_SCHEMA.md, and all files under state/.\n\
         Start with state/proposer_metadata.json, state/proposer_failure_summary.json, state/proposer_repair_hints.json, and state/proposer_examples.json.\n\
         Use shell/Python/JQ tools to inspect candidates, Pareto data, and rollout failures before editing proposal/manifest.json.\n\
         Propose exactly {} prompt candidates for generation {}.\n\
         Use only these target modules: {}.\n\
         Follow the Python GEPA workspace proposer style: diagnose the missing instruction type, reflect over wins and losses, then propose substantive prompt changes.\n\
         {proposal_policy}\n\
         At most one proposal may be conservative. The others must be very ambitious, task-specific updates that target named top failure clusters and are intended to substantially outperform the parent.\n\
         Shoot for large task-performance wins, not mild prompt polish. A safe clarification is only acceptable as the single conservative control.\n\
         Do not spend candidates on generic output-contract polish or parent paraphrases unless the dominant failures are output-format failures.\n\
         Across the requested proposals, explore genuinely different strategies (structural rewrite, few-shot examples, terse contract, label-table, role priming) rather than paraphrasing the seed or each other.\n\
         Write strict JSON to proposal/manifest.json using schema_version gepa_workspace_proposal_v3.\n\
         Include the required evidence block with reviewed files, candidate comparison, failure patterns, winning patterns, and example ids.\n\
         Do not print pseudo-tool calls. Use real file inspection and file editing.",
        input.config.gepa.proposals_per_generation,
        input.generation,
        input.config.candidate.target_modules.join(", ")
    ))
}

fn staleness_reviewer_instructions(input: &CodexStalenessReviewerInput<'_>) -> String {
    format!(
        "Read README.md, review/VERDICT_SCHEMA.md, and state/staleness_review_request.json. \
         Decide whether the stale FlashEvolve work should be accepted, discarded, or patched. \
         Use only these target modules when patching: {}. \
         Write the verdict as strict JSON to review/verdict.json. Do not print pseudo-tool calls; inspect files and edit the verdict file.",
        input.config.candidate.target_modules.join(", ")
    )
}

fn proposer_prompt_context(input: &CodexProposerInput<'_>) -> String {
    let rollouts = rollouts_read_model(input);
    let proposer_examples = proposer_examples_read_model(input);
    let proposer_failure_summary = proposer_failure_summary_read_model(input, &proposer_examples);
    let metadata = proposer_metadata_read_model(
        input,
        &rollouts,
        &proposer_examples,
        &proposer_failure_summary,
    );
    let proposer_model = metadata
        .pointer("/proposer/model")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let policy_model = metadata
        .pointer("/policy/model")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let frontier_size = metadata
        .get("frontier_size")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let candidate_count = metadata
        .get("candidate_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let rollout_count = metadata
        .get("rollout_row_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let loss_count = metadata
        .get("loss_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let win_count = metadata
        .get("win_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let top_failures = prompt_top_failures(&metadata);
    format!(
        "GEPA proposer context\n\
         run_id: {}\n\
         generation: {}\n\
         parent: {}\n\
         policy_model: {}\n\
         proposer_model: {}\n\
         target_levers: {}\n\
         mutable_levers: {}\n\
         proposals_requested: {}\n\
         minibatch_size: {}\n\
         candidates_seen: {}\n\
         frontier_size: {}\n\
         rollout_rows: {}\n\
         evidence: losses={} wins={}\n\
         workspace: {}\n\
         top_failures:\n\
         {}\n\
         read_first:\n\
         1. prompting_best_practices.md\n\
         2. state/proposer_metadata.json\n\
         3. state/proposer_failure_summary.json\n\
         4. state/proposer_repair_hints.json\n\
         5. state/proposer_examples.json\n\
         6. state/scores.json\n\
         7. state/parent_payload.json\n\
         8. state/candidate_deltas.json",
        input.config.run.run_id,
        input.generation,
        input.parent.candidate_id,
        policy_model,
        proposer_model,
        input.config.candidate.target_modules.join(", "),
        input.program.mutable_field_ids().join(", "),
        input.config.gepa.proposals_per_generation,
        input.config.gepa.minibatch_size,
        candidate_count,
        frontier_size,
        rollout_count,
        loss_count,
        win_count,
        input.workspace_dir.display(),
        top_failures
    )
}

fn prompt_top_failures(metadata: &Value) -> String {
    let items = metadata
        .get("top_failures")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if items.is_empty() {
        return "         - none observed yet".to_string();
    }
    items
        .into_iter()
        .take(3)
        .map(|item| {
            format!(
                "         - {}: {}",
                item.get("key").and_then(Value::as_str).unwrap_or("unknown"),
                item.get("count").and_then(Value::as_u64).unwrap_or(0)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn sandbox_policy_for_mode(mode: &str) -> Value {
    match mode {
        "danger-full-access" => json!({"type": "dangerFullAccess"}),
        "read-only" => {
            json!({"type": "readOnly", "access": {"type": "fullAccess"}, "networkAccess": true})
        }
        "workspace-write" => {
            json!({"type": "workspaceWrite", "readOnlyAccess": {"type": "fullAccess"}, "networkAccess": true})
        }
        _ => Value::String(mode.to_string()),
    }
}

fn read_manifest(workspace_dir: &Path) -> Result<Value> {
    let path = workspace_dir.join("proposal").join("manifest.json");
    let text = fs::read_to_string(&path).map_err(|source| OptimizerError::io(&path, source))?;
    if text.trim().is_empty() {
        return Err(OptimizerError::Proposer(format!(
            "codex app-server proposer wrote an empty manifest: {}",
            path.display()
        )));
    }
    match serde_json::from_str(&text) {
        Ok(value) => normalize_manifest_contract(value, &path),
        Err(original_error) => {
            let repaired = join_adjacent_json_strings(&text);
            if repaired == text {
                return Err(original_error.into());
            }
            let value = serde_json::from_str(&repaired).map_err(|_| original_error)?;
            write_text(&path, &repaired)?;
            normalize_manifest_contract(value, &path)
        }
    }
}

fn normalize_manifest_contract(mut manifest: Value, path: &Path) -> Result<Value> {
    let Some(object) = manifest.as_object_mut() else {
        return Ok(manifest);
    };
    let schema_version = object
        .get("schema_version")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    let has_proposals = object
        .get("proposals")
        .and_then(Value::as_array)
        .is_some_and(|items| !items.is_empty());
    if schema_version.is_empty() && has_proposals {
        object.insert(
            "schema_version".to_string(),
            Value::String(GEPA_WORKSPACE_PROPOSAL_SCHEMA_VERSION.to_string()),
        );
        write_json(path, &manifest)?;
    }
    Ok(manifest)
}

fn join_adjacent_json_strings(input: &str) -> String {
    let chars = input.chars().collect::<Vec<_>>();
    let mut out = String::with_capacity(input.len());
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;
    while index < chars.len() {
        let ch = chars[index];
        out.push(ch);
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
                let mut next = index + 1;
                while next < chars.len() && chars[next].is_whitespace() {
                    next += 1;
                }
                if next < chars.len() && chars[next] == '"' {
                    out.pop();
                    index = next;
                }
            }
        } else if ch == '"' {
            in_string = true;
        }
        index += 1;
    }
    out
}

fn proposals_from_manifest(manifest: &Value) -> Result<Value> {
    validate_manifest_contract(manifest)?;
    let proposals = manifest
        .get("proposals")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if proposals.is_empty() {
        return Err(OptimizerError::Proposer(
            "codex app-server proposer manifest contained no proposals".to_string(),
        ));
    }
    Ok(Value::Array(proposals))
}

fn validate_manifest_contract(manifest: &Value) -> Result<()> {
    let schema_version = manifest
        .get("schema_version")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if schema_version != GEPA_WORKSPACE_PROPOSAL_SCHEMA_VERSION {
        return Err(OptimizerError::Proposer(format!(
            "codex app-server proposer manifest schema_version={schema_version:?}; expected {GEPA_WORKSPACE_PROPOSAL_SCHEMA_VERSION}"
        )));
    }
    let evidence = manifest
        .get("evidence")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            OptimizerError::Proposer(
                "codex app-server proposer manifest omitted required evidence object".to_string(),
            )
        })?;
    let reviewed = evidence
        .get("reviewed_files")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect::<BTreeSet<_>>();
    if reviewed.is_empty() {
        return Err(OptimizerError::Proposer(
            "codex app-server proposer evidence reviewed_files is empty".to_string(),
        ));
    }
    for field in [
        "candidate_comparison",
        "failure_patterns",
        "winning_patterns",
        "example_ids_used",
    ] {
        let Some(value) = evidence.get(field) else {
            return Err(OptimizerError::Proposer(format!(
                "codex app-server proposer evidence missing {field}"
            )));
        };
        // Pattern fields may legitimately be empty when the observed evidence is
        // one-sided: all losses have no winning pattern, and all wins have no
        // failure pattern. Require the fields to be present, but do not force
        // content that the rollouts cannot ground.
        if matches!(field, "failure_patterns" | "winning_patterns") {
            continue;
        }
        let has_content = match value {
            Value::String(text) => !text.trim().is_empty(),
            Value::Array(items) => items.iter().any(|item| {
                item.as_str()
                    .map(|text| !text.trim().is_empty())
                    .unwrap_or(false)
            }),
            _ => false,
        };
        if !has_content {
            return Err(OptimizerError::Proposer(format!(
                "codex app-server proposer evidence field {field} is empty"
            )));
        }
    }
    Ok(())
}

fn manifest_evidence_warnings(
    input: &CodexProposerInput<'_>,
    manifest: &Value,
    proposals: &Value,
) -> Vec<String> {
    let mut warnings = Vec::new();
    let evidence = manifest.get("evidence").and_then(Value::as_object);
    let reviewed = evidence
        .and_then(|evidence| evidence.get("reviewed_files"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect::<BTreeSet<_>>();
    for required_file in [
        "state/proposer_metadata.json",
        "state/proposer_failure_summary.json",
        "state/proposer_repair_hints.json",
        "state/proposer_examples.json",
        "state/scores.json",
        "state/parent_payload.json",
    ] {
        if !reviewed.contains(required_file) {
            warnings.push(format!(
                "evidence reviewed_files did not include {required_file}"
            ));
        }
    }

    let cited_examples = evidence
        .and_then(|evidence| evidence.get("example_ids_used"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect::<BTreeSet<_>>();
    let proposer_examples = proposer_examples_read_model(input);
    let mut losing_examples = BTreeSet::new();
    let mut winning_examples = BTreeSet::new();
    for row in proposer_examples.as_array().into_iter().flatten() {
        let Some(example_id) = string_path(row, &["example_id"]) else {
            continue;
        };
        let reward = row.get("reward").and_then(Value::as_f64).unwrap_or(0.0);
        if reward >= 1.0 {
            winning_examples.insert(example_id);
        } else {
            losing_examples.insert(example_id);
        }
    }
    if !losing_examples.is_empty() && cited_examples.is_disjoint(&losing_examples) {
        warnings
            .push("evidence example_ids_used did not cite any losing rollout examples".to_string());
    }
    if !winning_examples.is_empty() && cited_examples.is_disjoint(&winning_examples) {
        warnings.push(
            "evidence example_ids_used did not cite any winning rollout examples".to_string(),
        );
    }

    let failure_summary = proposer_failure_summary_read_model(input, &proposer_examples);
    if let Some(dominant_failure_label) = dominant_failure_label(&failure_summary) {
        let evidence_text = evidence
            .map(|evidence| Value::Object(evidence.clone()).to_string())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let proposal_text = proposals.to_string().to_ascii_lowercase();
        let needle = dominant_failure_label.to_ascii_lowercase();
        if !evidence_text.contains(&needle) && !proposal_text.contains(&needle) {
            warnings.push(format!(
                "proposal did not mention dominant failure label {dominant_failure_label:?}"
            ));
        }
    }
    let proposal_text = proposals.to_string().to_ascii_lowercase();
    for row in proposer_examples.as_array().into_iter().flatten() {
        let Some(example_text) = string_path(row, &["text"]) else {
            continue;
        };
        let normalized = example_text.trim().to_ascii_lowercase();
        if normalized.len() >= 32 && proposal_text.contains(&normalized) {
            let example_id = string_path(row, &["example_id"]).unwrap_or_default();
            warnings.push(format!(
                "proposal appears to quote training example text from {example_id}; GEPA candidates should generalize rather than memorize exact queries"
            ));
            break;
        }
    }
    if literal_training_target_policy(input) == "forbid" {
        for row in proposer_examples.as_array().into_iter().flatten() {
            for field in ["expected", "prediction"] {
                let Some(literal) = string_path(row, &[field]) else {
                    continue;
                };
                let normalized = literal.trim().to_ascii_lowercase();
                if literal_target_is_specific(&normalized) && proposal_text.contains(&normalized) {
                    let example_id = string_path(row, &["example_id"]).unwrap_or_default();
                    warnings.push(format!(
                        "proposal appears to quote {field} target literal from {example_id} despite task policy forbidding literal training-target mappings"
                    ));
                    return warnings;
                }
            }
        }
    }
    warnings
}

fn dominant_failure_label(failure_summary: &Value) -> Option<String> {
    let key = failure_summary
        .get("label_confusions")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(|item| item.get("key"))
        .and_then(Value::as_str)?;
    let label = key
        .split(" -> ")
        .next()
        .unwrap_or_default()
        .trim()
        .to_string();
    if label.is_empty() || label == "unknown" {
        None
    } else {
        Some(label)
    }
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let text = serde_json::to_string_pretty(value)?;
    write_text(path, &format!("{text}\n"))
}

fn read_staleness_verdict_json(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).map_err(|source| OptimizerError::io(path, source))?;
    match serde_json::from_str::<Value>(&text) {
        Ok(value) => return Ok(value),
        Err(source) if !source.to_string().contains("trailing characters") => {
            return Err(OptimizerError::from(source));
        }
        Err(_) => {}
    }

    let mut values = Vec::new();
    for value in serde_json::Deserializer::from_str(&text).into_iter::<Value>() {
        values.push(value.map_err(OptimizerError::from)?);
    }
    values.into_iter().last().ok_or_else(|| {
        OptimizerError::Proposer(format!(
            "staleness reviewer verdict file {} did not contain a JSON object",
            path.display()
        ))
    })
}

fn write_text(path: &Path, text: &str) -> Result<()> {
    fs::write(path, text).map_err(|source| OptimizerError::io(path, source))
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    let value = value?.trim();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}
