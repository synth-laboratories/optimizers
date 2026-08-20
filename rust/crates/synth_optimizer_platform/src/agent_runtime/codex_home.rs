use std::collections::BTreeMap;
use std::env;
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::{
    resolve_chatgpt_codex_home_source, resolve_proposer_auth_launch_mode, McpAgentConfig,
    OptimizerError, ProposerAuthLaunchMode, ProposerConfig, Result,
};

/// Environment and cleanup state for a Codex app-server subprocess launch.
pub struct ProposerCodexLaunch {
    pub env_map: BTreeMap<String, String>,
    pub auth_home_to_cleanup: Option<PathBuf>,
    pub auth_home_refresh_source: Option<PathBuf>,
    pub codex_home_host_path: Option<PathBuf>,
    pub codex_home_workspace_relative_path: Option<PathBuf>,
}

pub fn prepare_proposer_codex_launch(
    proposer: &ProposerConfig,
    workspace_dir: &Path,
    model: &str,
    env_map: BTreeMap<String, String>,
) -> Result<ProposerCodexLaunch> {
    let proposer_api_key_env =
        non_empty(proposer.api_key_env.as_deref()).unwrap_or("OPENAI_API_KEY");
    let proposer_api_key = env::var(proposer_api_key_env)
        .ok()
        .filter(|api_key| !api_key.trim().is_empty());
    let launch_mode = resolve_proposer_auth_launch_mode(proposer, proposer_api_key.is_some())?;
    let mut env_map = env_map;
    let (
        auth_home_to_cleanup,
        auth_home_refresh_source,
        codex_home_host_path,
        codex_home_workspace_relative_path,
    ) = match launch_mode {
        ProposerAuthLaunchMode::ApiKey => {
            let api_key = proposer_api_key.ok_or_else(|| {
                OptimizerError::Proposer(format!(
                    "proposer.auth_mode = \"api_key\" requires non-empty {proposer_api_key_env}"
                ))
            })?;
            let codex_home_relative = PathBuf::from(".codex_api_key_home");
            let codex_home = workspace_dir.join(&codex_home_relative);
            // Codex speaks the Responses wire only (chat wire was removed upstream:
            // github.com/openai/codex/discussions/7782). Providers that serve only
            // chat-completions (NVIDIA direct) must use the chat_completions proposer
            // backend instead; OpenRouter proxies the Responses wire so it works here.
            prepare_api_key_codex_home(
                &codex_home,
                &proposer.provider,
                proposer.base_url.as_deref(),
                model,
                &api_key,
                proposer.model_context_window,
                proposer.model_auto_compact_token_limit,
            )?;
            env_map.insert("CODEX_HOME".to_string(), codex_home.display().to_string());
            // Codex reads OPENAI_API_KEY from the subprocess environment even when the
            // operator supplied the secret via another env var (for example OPENROUTER_API_KEY).
            env_map.insert("OPENAI_API_KEY".to_string(), api_key);
            (
                Some(codex_home.clone()),
                None,
                Some(codex_home),
                Some(codex_home_relative),
            )
        }
        ProposerAuthLaunchMode::Chatgpt => {
            let source = resolve_chatgpt_codex_home_source(proposer)?;
            let codex_home_relative = PathBuf::from(".codex_home");
            let codex_home = workspace_dir.join(&codex_home_relative);
            copy_codex_home(&source, &codex_home)?;
            env_map.insert("CODEX_HOME".to_string(), codex_home.display().to_string());
            // Keep ChatGPT-token launches hermetic when the hosting process also
            // carries an unrelated OpenAI API key for policies or other services.
            env_map.remove("OPENAI_API_KEY");
            (
                Some(codex_home.clone()),
                Some(source),
                Some(codex_home),
                Some(codex_home_relative),
            )
        }
    };
    if let Some(home) = &codex_home_host_path {
        append_mcp_servers(home, &proposer.mcp)?;
    }
    Ok(ProposerCodexLaunch {
        env_map,
        auth_home_to_cleanup,
        auth_home_refresh_source,
        codex_home_host_path,
        codex_home_workspace_relative_path,
    })
}

pub fn persist_refreshed_chatgpt_codex_auth(
    staged_codex_home: &Path,
    source_codex_home: &Path,
) -> Result<bool> {
    let staged_auth_path = staged_codex_home.join("auth.json");
    if !staged_auth_path.is_file() {
        return Ok(false);
    }
    let content = fs::read(&staged_auth_path)
        .map_err(|source| OptimizerError::io(&staged_auth_path, source))?;
    persist_refreshed_chatgpt_codex_auth_bytes(source_codex_home, &content)?;
    Ok(true)
}

pub fn persist_refreshed_chatgpt_codex_auth_bytes(
    source_codex_home: &Path,
    content: &[u8],
) -> Result<()> {
    validate_chatgpt_auth_json_bytes(content)?;
    fs::create_dir_all(source_codex_home)
        .map_err(|source| OptimizerError::io(source_codex_home, source))?;
    let auth_path = source_codex_home.join("auth.json");
    let tmp_path =
        source_codex_home.join(format!(".auth.json.tmp.{}", uuid::Uuid::new_v4().simple()));
    fs::write(&tmp_path, content).map_err(|source| OptimizerError::io(&tmp_path, source))?;
    #[cfg(unix)]
    fs::set_permissions(&tmp_path, fs::Permissions::from_mode(0o600))
        .map_err(|source| OptimizerError::io(&tmp_path, source))?;
    fs::rename(&tmp_path, &auth_path).map_err(|source| OptimizerError::io(&auth_path, source))
}

fn validate_chatgpt_auth_json_bytes(content: &[u8]) -> Result<()> {
    let value: Value = serde_json::from_slice(content).map_err(|source| {
        OptimizerError::Proposer(format!(
            "refreshed ChatGPT Codex auth.json is not valid JSON: {source}"
        ))
    })?;
    if !value.is_object() {
        return Err(OptimizerError::Proposer(
            "refreshed ChatGPT Codex auth.json must be a JSON object".to_string(),
        ));
    }
    if json_string_present(value.get("OPENAI_API_KEY")) {
        return Err(OptimizerError::Proposer(
            "refreshed ChatGPT Codex auth.json unexpectedly contains API-key auth".to_string(),
        ));
    }
    let tokens = value
        .get("tokens")
        .or_else(|| value.get("openai").and_then(|openai| openai.get("tokens")))
        .ok_or_else(|| {
            OptimizerError::Proposer(
                "refreshed ChatGPT Codex auth.json is missing token bundle".to_string(),
            )
        })?;
    for key in ["access_token", "id_token", "refresh_token", "account_id"] {
        if !json_string_present(tokens.get(key)) {
            return Err(OptimizerError::Proposer(format!(
                "refreshed ChatGPT Codex auth.json is missing tokens.{key}"
            )));
        }
    }
    Ok(())
}

fn json_string_present(value: Option<&Value>) -> bool {
    value
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
}

fn copy_codex_home(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination).map_err(|source| OptimizerError::io(destination, source))?;
    let mut copied_auth = false;
    for filename in [
        "auth.json",
        "installation_id",
        "version.json",
        "models_cache.json",
    ] {
        let source_file = source.join(filename);
        if source_file.is_file() {
            let destination_file = destination.join(filename);
            fs::copy(&source_file, &destination_file)
                .map_err(|copy_error| OptimizerError::io(destination_file, copy_error))?;
            if filename == "auth.json" {
                copied_auth = true;
            }
        }
    }
    if !copied_auth {
        return Err(OptimizerError::Proposer(format!(
            "Codex home {source:?} is missing auth.json; run `codex auth login` or fix \
             proposer.codex_home"
        )));
    }
    Ok(())
}

fn prepare_api_key_codex_home(
    destination: &Path,
    provider: &str,
    base_url: Option<&str>,
    model: &str,
    api_key: &str,
    model_context_window: Option<u64>,
    model_auto_compact_token_limit: Option<u64>,
) -> Result<()> {
    if destination.exists() {
        fs::remove_dir_all(destination)
            .map_err(|source| OptimizerError::io(destination, source))?;
    }
    fs::create_dir_all(destination).map_err(|source| OptimizerError::io(destination, source))?;
    let config_path = destination.join("config.toml");
    let provider_base_url = base_url.or_else(|| proposer_provider_default_base_url(provider));
    // Command-based auth, not `env_key`. Both authenticate, but OpenRouter's Codex
    // CLI guide is explicit that with a plain `env_key` "Codex won't fetch the
    // OpenRouter model catalog", so every non-OpenAI slug falls back to "Unknown
    // model" metadata; command-based auth is what triggers the catalog refresh.
    // Bad metadata means Codex sizes the turn from defaults rather than the model's
    // real context window. The command reads OPENAI_API_KEY because the launch path
    // already injects the resolved proposer secret into the subprocess under that
    // name regardless of which env var the operator supplied it in, so the key stays
    // out of config.toml.
    let provider_config = provider_base_url
        .map(|url| {
            format!(
                "model_provider = \"gepa_proposer\"\n\
                 \n\
                 [model_providers.gepa_proposer]\n\
                 name = \"GEPA proposer\"\n\
                 base_url = {url:?}\n\
                 wire_api = \"responses\"\n\
                 \n\
                 [model_providers.gepa_proposer.auth]\n\
                 command = \"sh\"\n\
                 args = [\"-c\", \"echo $OPENAI_API_KEY\"]\n\
                 \n"
            )
        })
        .unwrap_or_default();
    // Codex exposes no per-call output cap; the turn budget is governed by the
    // context window and the auto-compact threshold. Left unset, Codex falls back
    // to defaults sized for models it ships metadata for, which truncates or
    // compacts a large proposer turn against an unrecognised slug.
    let mut token_config = String::new();
    if let Some(window) = model_context_window {
        token_config.push_str(&format!("model_context_window = {window}\n"));
    }
    if let Some(limit) = model_auto_compact_token_limit {
        token_config.push_str(&format!("model_auto_compact_token_limit = {limit}\n"));
    }
    write_text(
        &config_path,
        &format!(
            "model = {model:?}\n\
             preferred_auth_method = \"apikey\"\n\
             {token_config}\
             {provider_config}\
             [features]\n\
             apps = false\n\
             browser_use = false\n\
             browser_use_external = false\n\
             computer_use = false\n\
             image_generation = false\n\
             in_app_browser = false\n\
             multi_agent = false\n\
             plugins = false\n\
             skill_mcp_dependency_install = false\n\
             tool_suggest = false\n\
             workspace_dependencies = false\n"
        ),
    )?;
    let auth_path = destination.join("auth.json");
    let encoded_key = serde_json::to_string(api_key)?;
    write_text(
        &auth_path,
        &format!("{{\"OPENAI_API_KEY\":{encoded_key}}}\n"),
    )?;
    #[cfg(unix)]
    fs::set_permissions(&auth_path, fs::Permissions::from_mode(0o600))
        .map_err(|source| OptimizerError::io(&auth_path, source))?;
    Ok(())
}

fn proposer_provider_default_base_url(provider: &str) -> Option<&'static str> {
    match provider.trim().to_ascii_lowercase().as_str() {
        "openrouter" => Some("https://openrouter.ai/api/v1"),
        "deepseek" => Some("https://api.deepseek.com"),
        "nvidia" => Some("https://integrate.api.nvidia.com/v1"),
        _ => None,
    }
}

fn append_mcp_servers(codex_home: &Path, mcp: &McpAgentConfig) -> Result<()> {
    if !mcp.enabled {
        return Ok(());
    }
    let command = mcp.command.as_deref().unwrap_or("").trim();
    let server = mcp
        .server
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("gepa_external");
    if command.is_empty() {
        return Err(OptimizerError::Config(
            "proposer.mcp.command is required when mcp.enabled".to_string(),
        ));
    }
    let config_path = codex_home.join("config.toml");
    let mut body = if config_path.exists() {
        fs::read_to_string(&config_path).map_err(|source| OptimizerError::io(&config_path, source))?
    } else {
        String::new()
    };
    let header = format!("[mcp_servers.{server}]");
    if !body.contains(&header) {
        let mut parts = command.split_whitespace();
        let cmd = parts.next().unwrap_or(server);
        let args: Vec<String> = parts.map(|part| format!("{part:?}")).collect();
        body.push('\n');
        body.push_str(&header);
        body.push('\n');
        body.push_str(&format!("command = {cmd:?}\n"));
        body.push_str(&format!("args = [{}]\n", args.join(", ")));
        write_text(&config_path, &body)?;
    }
    if let Some(state_dir) = codex_home.parent().map(|parent| parent.join("state")) {
        if state_dir.is_dir() {
            let receipt = state_dir.join("mcp_codex_receipt.json");
            let payload = serde_json::json!({
                "schema_version": "gepa_mcp_codex_receipt.v1",
                "mcp_in_codex_config": true,
                "server": server,
                "command": command,
                "config_path": config_path.display().to_string(),
            });
            fs::write(&receipt, serde_json::to_vec_pretty(&payload)?)
                .map_err(|source| OptimizerError::io(&receipt, source))?;
        }
    }
    Ok(())
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
