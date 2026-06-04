use std::collections::BTreeMap;
use std::env;
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use crate::{
    resolve_chatgpt_codex_home_source, resolve_proposer_auth_launch_mode, OptimizerError,
    ProposerAuthLaunchMode, ProposerConfig, Result,
};

/// Environment and cleanup state for a Codex app-server subprocess launch.
pub struct ProposerCodexLaunch {
    pub env_map: BTreeMap<String, String>,
    pub auth_home_to_cleanup: Option<PathBuf>,
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
    let (auth_home_to_cleanup, codex_home_host_path, codex_home_workspace_relative_path) =
        match launch_mode {
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
                )?;
                env_map.insert("CODEX_HOME".to_string(), codex_home.display().to_string());
                // Codex reads OPENAI_API_KEY from the subprocess environment even when the
                // operator supplied the secret via another env var (for example OPENROUTER_API_KEY).
                env_map.insert("OPENAI_API_KEY".to_string(), api_key);
                (
                    Some(codex_home.clone()),
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
                (
                    Some(codex_home.clone()),
                    Some(codex_home),
                    Some(codex_home_relative),
                )
            }
        };
    Ok(ProposerCodexLaunch {
        env_map,
        auth_home_to_cleanup,
        codex_home_host_path,
        codex_home_workspace_relative_path,
    })
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
) -> Result<()> {
    if destination.exists() {
        fs::remove_dir_all(destination)
            .map_err(|source| OptimizerError::io(destination, source))?;
    }
    fs::create_dir_all(destination).map_err(|source| OptimizerError::io(destination, source))?;
    let config_path = destination.join("config.toml");
    let provider_base_url = base_url.or_else(|| proposer_provider_default_base_url(provider));
    let provider_config = provider_base_url
        .map(|url| {
            format!(
                "model_provider = \"gepa_proposer\"\n\
                 \n\
                 [model_providers.gepa_proposer]\n\
                 name = \"GEPA proposer\"\n\
                 base_url = {url:?}\n\
                 env_key = \"OPENAI_API_KEY\"\n\
                 wire_api = \"responses\"\n\
                 \n"
            )
        })
        .unwrap_or_default();
    write_text(
        &config_path,
        &format!(
            "model = {model:?}\n\
             preferred_auth_method = \"apikey\"\n\
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
