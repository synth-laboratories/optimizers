use std::collections::BTreeMap;
use std::env;
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::{OptimizerError, Result};

use super::app_server::{CodexAppServerClient, CodexAppServerProcessLaunch};
use super::codex_home::prepare_proposer_codex_launch;
use super::limits;
use super::session::{
    run_codex_jsonrpc_turn, AgentRuntimeSubstrate, AgentTurnOutcome, CodexTurnRequest,
};
use super::supervisor::SupervisorReceipt;

pub struct DockerCodexSubstrate;

impl AgentRuntimeSubstrate for DockerCodexSubstrate {
    fn run_codex_turn(&self, request: CodexTurnRequest<'_>) -> Result<AgentTurnOutcome> {
        run_docker_codex_turn(request)
    }
}

fn run_docker_codex_turn(request: CodexTurnRequest<'_>) -> Result<AgentTurnOutcome> {
    let docker = request.proposer.docker.as_ref().ok_or_else(|| {
        OptimizerError::Config(
            "proposer.runtime_substrate = \"docker\" requires [proposer.docker]".to_string(),
        )
    })?;
    let image = docker
        .image
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            OptimizerError::Config(
                "proposer.runtime_substrate = \"docker\" requires [proposer.docker].image"
                    .to_string(),
            )
        })?
        .to_string();
    let workspace_mount_path = docker.workspace_mount_path.clone();
    let network = docker.network.clone();
    let extra_env = docker.extra_env.clone();
    check_docker_available()?;

    let original_workspace = fs::canonicalize(request.workspace_dir)
        .map_err(|source| OptimizerError::io(request.workspace_dir, source))?;
    let staged_workspace = stage_workspace(&original_workspace, request.run_id)?;
    let staged_workspace_result = run_docker_with_staged_workspace(
        request,
        &image,
        &workspace_mount_path,
        &network,
        &extra_env,
        &staged_workspace,
    );
    let sync_result = sync_workspace_back(&staged_workspace, &original_workspace);
    let cleanup_result = cleanup_staged_workspace(&staged_workspace);
    match (staged_workspace_result, sync_result, cleanup_result) {
        (Ok(mut outcome), Ok(()), Ok(())) => {
            if let Some(receipt) = outcome.supervisor_receipt.as_mut() {
                receipt.cleanup_status = "cleaned".to_string();
            }
            Ok(outcome)
        }
        (Err(error), Ok(()), Ok(())) => Err(error),
        (result, sync, cleanup) => Err(OptimizerError::Proposer(format!(
            "docker proposer workspace cleanup failed: run_result={}; sync_result={}; cleanup_result={}",
            result_status(&result),
            result_status(&sync),
            result_status(&cleanup)
        ))),
    }
}

fn run_docker_with_staged_workspace(
    request: CodexTurnRequest<'_>,
    image: &str,
    workspace_mount_path: &str,
    network: &str,
    extra_env: &BTreeMap<String, String>,
    staged_workspace: &Path,
) -> Result<AgentTurnOutcome> {
    write_entrypoint(staged_workspace)?;
    let host_env = env::vars().collect::<BTreeMap<_, _>>();
    let launch_state =
        prepare_proposer_codex_launch(request.proposer, staged_workspace, request.model, host_env)?;
    let mut docker_process_env = env::vars().collect::<BTreeMap<_, _>>();
    let container_name = container_name(request.run_id);
    let mut docker_args = vec![
        "docker".to_string(),
        "run".to_string(),
        "--rm".to_string(),
        "-i".to_string(),
        "--name".to_string(),
        container_name.clone(),
        "--network".to_string(),
        network.to_string(),
    ];
    if codex_sandbox_requires_linux_namespaces(request.proposer) {
        docker_args.extend([
            "--cap-add".to_string(),
            "SYS_ADMIN".to_string(),
            "--security-opt".to_string(),
            "seccomp=unconfined".to_string(),
        ]);
    }
    docker_args.extend([
        "-v".to_string(),
        format!("{}:{}", staged_workspace.display(), workspace_mount_path),
        "-w".to_string(),
        workspace_mount_path.to_string(),
    ]);
    let container_codex_home = container_codex_home_path(&launch_state, workspace_mount_path)?;
    docker_args.push("-e".to_string());
    docker_args.push(format!("CODEX_HOME={container_codex_home}"));
    docker_args.push("-e".to_string());
    docker_args.push(format!("SYNTH_WORKSPACE={workspace_mount_path}"));
    if let Some(api_key) = launch_state.env_map.get("OPENAI_API_KEY") {
        docker_process_env.insert("OPENAI_API_KEY".to_string(), api_key.clone());
        docker_args.push("-e".to_string());
        docker_args.push("OPENAI_API_KEY".to_string());
    }
    add_extra_env_refs(extra_env, &mut docker_process_env, &mut docker_args)?;
    docker_args.push(image.to_string());
    docker_args.push(format!(
        "{}/.codex_app_server_entrypoint.sh",
        workspace_mount_path.trim_end_matches('/')
    ));
    docker_args.extend(inner_codex_command(request.proposer));

    let process_label = format!(
        "docker codex app-server container={} image={image}",
        container_name
    );
    let client = CodexAppServerClient::start_process(CodexAppServerProcessLaunch {
        command: docker_args.clone(),
        current_dir: staged_workspace.to_path_buf(),
        env_map: docker_process_env,
        auth_home_to_cleanup: launch_state.auth_home_to_cleanup,
        process_label,
        execution_mode: "local_process".to_string(),
    })?;
    let receipt = SupervisorReceipt {
        substrate: "docker".to_string(),
        process_id: Some(client.process_id()),
        container_name: Some(container_name),
        image: Some(image.to_string()),
        staging_dir: Some(staged_workspace.display().to_string()),
        workspace_mount_path: Some(workspace_mount_path.to_string()),
        cleanup_status: "pending".to_string(),
    };
    eprintln!(
        "[gepa-proposer] docker substrate started run_id={} container={} image={} staging_dir={}",
        request.run_id,
        receipt.container_name.as_deref().unwrap_or("unknown"),
        image,
        staged_workspace.display()
    );
    run_codex_jsonrpc_turn(client, request, Some(receipt))
}

fn check_docker_available() -> Result<()> {
    let output = Command::new("docker")
        .arg("info")
        .arg("--format")
        .arg("{{.ServerVersion}}")
        .stdin(Stdio::null())
        .output()
        .map_err(|source| {
            OptimizerError::Proposer(format!(
                "docker proposer preflight failed to run `docker info`: {source}"
            ))
        })?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(OptimizerError::Proposer(format!(
        "docker proposer preflight failed: `docker info` exited with {}; stderr={}",
        output.status,
        stderr.trim()
    )))
}

fn stage_workspace(original_workspace: &Path, run_id: &str) -> Result<PathBuf> {
    let staging_root = docker_workspace_root()?;
    let staging_dir = staging_root.join(format!(
        "{}-{}",
        safe_fragment(run_id),
        uuid::Uuid::new_v4().simple()
    ));
    fs::create_dir_all(&staging_dir).map_err(|source| OptimizerError::io(&staging_dir, source))?;
    copy_dir_contents(original_workspace, &staging_dir, true)?;
    Ok(staging_dir)
}

fn sync_workspace_back(staged_workspace: &Path, original_workspace: &Path) -> Result<()> {
    copy_dir_contents(staged_workspace, original_workspace, true)
}

fn cleanup_staged_workspace(staged_workspace: &Path) -> Result<()> {
    if staged_workspace.exists() {
        fs::remove_dir_all(staged_workspace)
            .map_err(|source| OptimizerError::io(staged_workspace, source))?;
    }
    Ok(())
}

fn docker_workspace_root() -> Result<PathBuf> {
    let home = env::var_os("HOME").ok_or_else(|| {
        OptimizerError::Proposer(
            "docker proposer staging requires HOME for ~/.cache/synth-gepa-docker-workspaces"
                .to_string(),
        )
    })?;
    let root = PathBuf::from(home).join(limits::DOCKER_WORKSPACE_CACHE_DIR);
    fs::create_dir_all(&root).map_err(|source| OptimizerError::io(&root, source))?;
    Ok(root)
}

fn write_entrypoint(workspace: &Path) -> Result<()> {
    let path = workspace.join(".codex_app_server_entrypoint.sh");
    fs::write(
        &path,
        "#!/bin/sh\nset -eu\ncd \"${SYNTH_WORKSPACE:-/workspace}\"\nexec \"$@\"\n",
    )
    .map_err(|source| OptimizerError::io(&path, source))?;
    #[cfg(unix)]
    {
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755))
            .map_err(|source| OptimizerError::io(&path, source))?;
    }
    Ok(())
}

fn container_codex_home_path(
    launch_state: &super::codex_home::ProposerCodexLaunch,
    workspace_mount_path: &str,
) -> Result<String> {
    let relative = launch_state
        .codex_home_workspace_relative_path
        .as_ref()
        .ok_or_else(|| {
            OptimizerError::Proposer(
                "docker proposer auth preparation did not produce a workspace-relative CODEX_HOME"
                    .to_string(),
            )
        })?;
    Ok(format!(
        "{}/{}",
        workspace_mount_path.trim_end_matches('/'),
        relative.display()
    ))
}

fn add_extra_env_refs(
    extra_env: &BTreeMap<String, String>,
    docker_process_env: &mut BTreeMap<String, String>,
    docker_args: &mut Vec<String>,
) -> Result<()> {
    for (container_key, host_key) in extra_env {
        let value = env::var(host_key).map_err(|source| {
            OptimizerError::Config(format!(
                "proposer.docker.extra_env maps {container_key} to host env {host_key}, but it is unavailable: {source}"
            ))
        })?;
        if value.trim().is_empty() {
            return Err(OptimizerError::Config(format!(
                "proposer.docker.extra_env maps {container_key} to host env {host_key}, but it is empty"
            )));
        }
        docker_process_env.insert(container_key.clone(), value);
        docker_args.push("-e".to_string());
        docker_args.push(container_key.clone());
    }
    Ok(())
}

fn inner_codex_command(proposer: &crate::ProposerConfig) -> Vec<String> {
    if proposer.command.is_empty() {
        vec!["codex".to_string(), "app-server".to_string()]
    } else {
        proposer.command.clone()
    }
}

fn codex_sandbox_requires_linux_namespaces(proposer: &crate::ProposerConfig) -> bool {
    match proposer
        .sandbox_mode
        .as_deref()
        .map(str::trim)
        .filter(|mode| !mode.is_empty())
    {
        Some("danger-full-access") | None => false,
        Some(_) => true,
    }
}

fn copy_dir_contents(source: &Path, destination: &Path, exclude_runtime_auth: bool) -> Result<()> {
    fs::create_dir_all(destination).map_err(|source| OptimizerError::io(destination, source))?;
    for entry in
        fs::read_dir(source).map_err(|read_error| OptimizerError::io(source, read_error))?
    {
        let entry = entry.map_err(|read_error| OptimizerError::io(source, read_error))?;
        let name = entry.file_name();
        if exclude_runtime_auth && excluded_workspace_entry(&name.to_string_lossy()) {
            continue;
        }
        let source_path = entry.path();
        let destination_path = destination.join(&name);
        let metadata = fs::metadata(&source_path)
            .map_err(|metadata_error| OptimizerError::io(&source_path, metadata_error))?;
        if metadata.is_dir() {
            copy_dir_contents(&source_path, &destination_path, exclude_runtime_auth)?;
        } else if metadata.is_file() {
            if let Some(parent) = destination_path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|create_error| OptimizerError::io(parent, create_error))?;
            }
            fs::copy(&source_path, &destination_path)
                .map_err(|copy_error| OptimizerError::io(&destination_path, copy_error))?;
        }
    }
    Ok(())
}

fn excluded_workspace_entry(name: &str) -> bool {
    matches!(
        name,
        ".codex_api_key_home" | ".codex_home" | ".codex_app_server_entrypoint.sh"
    )
}

fn container_name(run_id: &str) -> String {
    format!(
        "synth-gepa-proposer-{}-{}",
        safe_fragment(run_id),
        &uuid::Uuid::new_v4().simple().to_string()[..8]
    )
}

fn safe_fragment(value: &str) -> String {
    let mut output = value
        .chars()
        .filter_map(|ch| {
            if ch.is_ascii_alphanumeric() {
                Some(ch.to_ascii_lowercase())
            } else if matches!(ch, '-' | '_') {
                Some('-')
            } else {
                None
            }
        })
        .take(48)
        .collect::<String>();
    if output.is_empty() {
        output = "run".to_string();
    }
    output
}

fn result_status<T>(result: &Result<T>) -> String {
    match result {
        Ok(_) => "ok".to_string(),
        Err(error) => format!("err({error})"),
    }
}
