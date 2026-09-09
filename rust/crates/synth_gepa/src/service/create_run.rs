use super::*;
use std::sync::{Mutex, OnceLock};

/// `set_var` is process-global; these tests must not interleave.
fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn test_runtime() -> GepaServiceRuntime {
    let db_path = std::env::temp_dir().join(format!(
        "synth_gepa_create_run_{}_{}.sqlite",
        std::process::id(),
        now_millis()
    ));
    GepaServiceRuntime {
        config: GepaServiceConfig::new(db_path, "127.0.0.1:0"),
        scheduler: ServiceSchedulerSignal::new(),
        service_url: "http://127.0.0.1:0".to_string(),
        started_at: crate::rfc3339_now(),
    }
}

fn run_request() -> GepaServiceRunRequest {
    serde_json::from_value(json!({
        "container_url": "http://127.0.0.1:9/never-reached",
        "policy": {
            "provider": "openai",
            "model": "gpt-4.1-nano",
            "credentials": {"resolver": "env", "env_var": "OPENAI_API_KEY"},
        },
        "proposer": {
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "credentials": {"resolver": "env", "env_var": "OPENAI_API_KEY"},
        },
        "taskset": {"train_ids": ["t1"], "heldout_ids": ["t2"]},
        "task_pools": {
            "pareto": ["t1"],
            "minibatch": ["t1"],
            "reflection": ["t1"],
            "heldout": ["t2"],
        },
    }))
    .expect("run request parses")
}

#[test]
fn forbidden_names_are_exactly_the_two_override_prefixes() {
    let found = forbidden_runtime_env_vars_in([
        "SYNTH_OPTIMIZERS_PROPOSER_MODEL",
        "GEPA_PLATFORM_RUN_ID",
        "SYNTH_BACKEND_URL",
        "SYNTH_WORKSHOP_INSTANCE_ID",
        "GEPA_HOME",
        "PATH",
    ]);
    assert_eq!(
        found,
        vec![
            "GEPA_PLATFORM_RUN_ID".to_string(),
            "SYNTH_OPTIMIZERS_PROPOSER_MODEL".to_string(),
        ]
    );
}

#[test]
fn create_run_refuses_a_service_env_that_can_override_the_config() {
    let _guard = env_guard();
    std::env::set_var("SYNTH_OPTIMIZERS_PROPOSER_MODEL", "model-from-env");
    let runtime = test_runtime();
    let error = create_run(&runtime, run_request(), None, "sha".to_string())
        .expect_err("admission must refuse");
    std::env::remove_var("SYNTH_OPTIMIZERS_PROPOSER_MODEL");

    let message = error.to_string();
    assert!(
        message.contains("SYNTH_OPTIMIZERS_PROPOSER_MODEL"),
        "the refusal names the variable to unset: {message}"
    );
    let response = optimizer_error_response(error);
    assert_eq!(response.status, 422);
    let body: Value = serde_json::from_slice(&response.body).expect("json body");
    assert_eq!(body["error"]["code"], "invalid_config");
    assert!(
        !body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("model-from-env"),
        "the value is never echoed back"
    );
    assert!(
        !runtime.config.db_path.exists(),
        "admission refuses before it opens the workspace"
    );
}

#[test]
fn a_clean_service_env_gets_past_the_guard() {
    let _guard = env_guard();
    let stashed: Vec<(String, String)> = std::env::vars()
        .filter(|(name, _)| {
            FORBIDDEN_RUNTIME_ENV_PREFIXES
                .iter()
                .any(|prefix| name.starts_with(prefix))
        })
        .collect();
    for (name, _) in &stashed {
        std::env::remove_var(name);
    }
    let runtime = test_runtime();
    // The container is unreachable, so this fails — but on the contract
    // handshake, not on the env guard.
    let error = create_run(&runtime, run_request(), None, "sha".to_string())
        .expect_err("the fake container is unreachable");
    for (name, value) in stashed {
        std::env::set_var(name, value);
    }
    assert!(
        !error.to_string().contains("run-config overrides"),
        "a clean environment must not trip the guard: {error}"
    );
    std::fs::remove_file(&runtime.config.db_path).ok();
}
