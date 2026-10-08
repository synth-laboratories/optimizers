use std::collections::BTreeMap;
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::{
    OptimizerError, ProposerAuthLaunchMode, ProposerConfig, Result,
    resolve_chatgpt_codex_home_source, resolve_proposer_auth_launch_mode,
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
    if proposer.api_key_env.is_some() {
        return Err(OptimizerError::Proposer(
            "Codex accepts proposer.gateway_session, never a provider key environment reference"
                .into(),
        ));
    }
    let gateway_environment = proposer
        .gateway_session
        .as_deref()
        .map(|handle| {
            let handle = synth_gateway_client::SessionHandle::parse(handle)
                .map_err(|error| OptimizerError::Proposer(error.to_string()))?;
            synth_gateway_client::session(&handle)
                .and_then(|session| session.sandbox_environment(model))
                .map_err(|error| OptimizerError::Proposer(format!("Codex gateway: {error}")))
        })
        .transpose()?;
    let launch_mode = resolve_proposer_auth_launch_mode(proposer, gateway_environment.is_some())?;
    let mut env_map = env_map;
    for key in [
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "SYNTH_API_KEY",
    ] {
        env_map.remove(key);
    }
    let (
        auth_home_to_cleanup,
        auth_home_refresh_source,
        codex_home_host_path,
        codex_home_workspace_relative_path,
    ) = match launch_mode {
        ProposerAuthLaunchMode::ApiKey => {
            let environment = gateway_environment.ok_or_else(|| {
                OptimizerError::Proposer("Codex gateway session is unbound".into())
            })?;
            let codex_home_relative = PathBuf::from(".codex_gateway_home");
            let codex_home = workspace_dir.join(&codex_home_relative);
            prepare_gateway_codex_home(
                &codex_home,
                &environment["SYNTH_GATEWAY_SANDBOX_BASE_URL"],
                model,
            )?;
            env_map.insert("CODEX_HOME".to_string(), codex_home.display().to_string());
            env_map.extend(environment);
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
    // A staged home is disposable process state. Reusing it can retain a cache
    // written by a different Codex binary and makes a retry nondeterministic.
    if destination.exists() {
        fs::remove_dir_all(destination)
            .map_err(|remove_error| OptimizerError::io(destination, remove_error))?;
    }
    fs::create_dir_all(destination).map_err(|source| OptimizerError::io(destination, source))?;
    let mut copied_auth = false;
    // models_cache.json is intentionally not copied. Its schema belongs to the
    // exact app-server binary being launched; Codex regenerates it. Copying a
    // cache from another version can make model loading fail after the turn has
    // already started (for example when a newly required field is absent).
    for filename in ["auth.json", "installation_id", "version.json"] {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatgpt_home_staging_is_clean_and_never_copies_model_cache() {
        let root = std::env::temp_dir().join(format!(
            "synth-codex-home-staging-{}",
            uuid::Uuid::new_v4().simple()
        ));
        let source = root.join("source");
        let destination = root.join("destination");
        fs::create_dir_all(&source).unwrap();
        fs::create_dir_all(&destination).unwrap();
        fs::write(source.join("auth.json"), b"{}\n").unwrap();
        fs::write(source.join("installation_id"), b"installation\n").unwrap();
        fs::write(
            source.join("models_cache.json"),
            br#"{"models":[{"slug":"stale","missing_new_fields":true}]}"#,
        )
        .unwrap();
        fs::write(
            destination.join("models_cache.json"),
            b"stale retry cache\n",
        )
        .unwrap();
        fs::write(destination.join("unrelated-runtime-state"), b"stale\n").unwrap();

        copy_codex_home(&source, &destination).unwrap();

        assert_eq!(fs::read(destination.join("auth.json")).unwrap(), b"{}\n");
        assert_eq!(
            fs::read(destination.join("installation_id")).unwrap(),
            b"installation\n"
        );
        assert!(!destination.join("models_cache.json").exists());
        assert!(!destination.join("unrelated-runtime-state").exists());
        assert!(source.join("models_cache.json").exists());
        fs::remove_dir_all(root).unwrap();
    }
}

fn prepare_gateway_codex_home(destination: &Path, base_url: &str, model: &str) -> Result<()> {
    if destination.exists() {
        fs::remove_dir_all(destination)
            .map_err(|source| OptimizerError::io(destination, source))?;
    }
    fs::create_dir_all(destination).map_err(|source| OptimizerError::io(destination, source))?;
    write_text(
        &destination.join("config.toml"),
        &format!(
            "model = {model:?}\nmodel_provider = \"synth_gateway\"\n[model_providers.synth_gateway]\nname = \"Synth inference gateway\"\nbase_url = {base_url:?}\nenv_key = \"SYNTH_GATEWAY_SESSION_TOKEN\"\nwire_api = \"responses\"\nrequires_openai_auth = false\n[features]\napps = false\nbrowser_use = false\nbrowser_use_external = false\ncomputer_use = false\nimage_generation = false\nin_app_browser = false\nmulti_agent = false\nplugins = false\nskill_mcp_dependency_install = false\ntool_suggest = false\nworkspace_dependencies = false\n"
        ),
    )?;
    Ok(())
}

fn write_text(path: &Path, text: &str) -> Result<()> {
    fs::write(path, text).map_err(|source| OptimizerError::io(path, source))
}
