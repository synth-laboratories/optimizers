//! Evaluate one real local-container request through the optimizer's owned launcher.
//! Session tokens are runtime environment only; stdin contains a handle and task input.
use std::io::{self, Read};
use serde::Deserialize;
use serde_json::Value;
use synth_optimizer_platform::{ContainerConfig, PolicyConfig, ManagedContainerProcess, ContainerClient};

#[derive(Deserialize)]
struct Evaluation {
    container: ContainerConfig,
    policy: PolicyConfig,
    request: Value,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let mut evaluation: Evaluation = serde_json::from_str(&input)?;
    let _process = ManagedContainerProcess::maybe_start_with_gateway(&evaluation.container, &evaluation.policy)?;
    evaluation.request["policy"]["config"] = evaluation.policy.sandbox_wire_config()?;
    evaluation.request["policy"]["config"]["agent"] = serde_json::json!("codex");
    evaluation.request["policy"]["config"]["timeout"] = serde_json::json!(120);
    let url = evaluation.container.url.as_deref().ok_or("container.url missing")?;
    let client = ContainerClient::with_headers_bearer_env_and_timeout(url,
        evaluation.container.headers, evaluation.container.auth_bearer_env.as_deref(), Some(180.0))?;
    let response = client.rollout(&evaluation.request)?;
    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}
