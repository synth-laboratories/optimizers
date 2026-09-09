use super::*;
use std::sync::{Mutex, OnceLock};

/// `set_var` is process-global; these tests must not interleave.
fn env_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const MINIMAL_TOML: &str = r#"
[run]
run_id = "gepa_from_toml"
output_dir = "runs/from_toml"

[container]
url = "http://127.0.0.1:8099"

[taskset]
train_ids = ["t1"]
heldout_ids = ["t2"]

[candidate]
target_modules = ["classify"]

[gepa.task_pools]
pareto = ["t1"]
minibatch = ["t1"]
reflection = ["t1"]
heldout = ["t2"]

[policy]
proxy_mode = "proxy_only"

[proposer]
model = "model-from-toml"
reasoning_effort = "low"

[cache]
namespace = "namespace-from-toml"

[jesterky_workflow]
spec = "specs/from_toml.yaml"
"#;

fn write_config(dir: &Path) -> PathBuf {
    let path = dir.join("gepa.toml");
    fs::write(&path, MINIMAL_TOML).expect("write config");
    path
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "synth-optimizers-config-{name}-{}",
        uuid::Uuid::new_v4().simple()
    ));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

#[test]
fn env_cannot_override_the_loaded_config() {
    let _guard = env_guard();
    let dir = temp_dir("env-override");
    let path = write_config(&dir);

    // Every one of these used to change the run. None of them may now.
    let overrides = [
        ("SYNTH_OPTIMIZERS_PROPOSER_MODEL", "model-from-env"),
        ("GEPA_PLATFORM_PROPOSER_MODEL", "model-from-env-alias"),
        ("SYNTH_OPTIMIZERS_RUN_ID", "run-from-env"),
        ("SYNTH_OPTIMIZERS_CACHE_NAMESPACE", "namespace-from-env"),
        ("SYNTH_OPTIMIZERS_OUTPUT_DIR", "/tmp/output-from-env"),
        ("SYNTH_OPTIMIZERS_PROPOSER_REASONING_EFFORT", "high"),
    ];
    for (name, value) in overrides {
        std::env::set_var(name, value);
    }

    let config = SynthOptimizerConfig::from_toml_file(&path).expect("load config");

    for (name, _) in overrides {
        std::env::remove_var(name);
    }

    assert_eq!(config.proposer.model.as_deref(), Some("model-from-toml"));
    assert_eq!(config.run.run_id, "gepa_from_toml");
    assert_eq!(
        config.cache.namespace.as_deref(),
        Some("namespace-from-toml")
    );
    assert_eq!(config.proposer.reasoning_effort.as_deref(), Some("low"));
    assert!(
        config.run.output_dir.starts_with(&dir),
        "output_dir came from the TOML, resolved against its own directory: {}",
        config.run.output_dir.display()
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn jesterky_spec_resolves_against_the_config_directory() {
    let dir = temp_dir("jesterky-spec");
    let path = write_config(&dir);
    let config = SynthOptimizerConfig::from_toml_file(&path).expect("load config");
    assert_eq!(
        PathBuf::from(&config.jesterky_workflow.spec),
        dir.join("specs/from_toml.yaml"),
        "spec must be absolute against the TOML directory, never a developer checkout"
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn only_one_backend_url_name_is_read() {
    let _guard = env_guard();
    let aliases = [
        "SYNTH_BACKEND_URL_OVERRIDE",
        "SYNTH_API_URL",
        "DEV_SYNTH_BACKEND_URL",
        "DEV_BACKEND_URL",
        "PROD_SYNTH_BACKEND_URL",
        "PROD_BACKEND_URL",
        "BACKEND_URL",
    ];
    let previous = std::env::var(BACKEND_BASE_URL_ENV).ok();
    std::env::remove_var(BACKEND_BASE_URL_ENV);
    for alias in aliases {
        std::env::set_var(alias, "https://alias.invalid");
    }
    assert_eq!(resolve_backend_base_url_from_env(), None);

    std::env::set_var(BACKEND_BASE_URL_ENV, "https://backend.invalid");
    assert_eq!(
        resolve_backend_base_url_from_env().as_deref(),
        Some("https://backend.invalid")
    );

    for alias in aliases {
        std::env::remove_var(alias);
    }
    match previous {
        Some(value) => std::env::set_var(BACKEND_BASE_URL_ENV, value),
        None => std::env::remove_var(BACKEND_BASE_URL_ENV),
    }
}

/// The deleted override layer must not come back by any name.
#[test]
fn no_env_override_helper_survives_in_the_workspace() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("repo root")
        .join("rust");
    let mut offenders = Vec::new();
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name == "target") {
                    continue;
                }
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let text = fs::read_to_string(&path).unwrap_or_default();
                // Split so this file is not its own offender.
                if text.contains(concat!("read_env", "_override")) {
                    offenders.push(path.display().to_string());
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "the env-override helper is deleted; a run's config is sealed at \
             admission: {offenders:?}"
    );
}

/// Production `config.rs` may read at most two variables. It reads one: the
/// single backend URL name. `GEPA_HOME` and the Workshop instance id are
/// read elsewhere. Test code below the `#[cfg(test)]` line is not counted;
/// the needle is split so this file is not its own offender.
#[test]
fn config_reads_at_most_two_env_vars() {
    let source = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/config.rs"))
        .expect("read config.rs");
    let production = source
        .split_once("\n#[cfg(test)]")
        .map(|(before, _)| before)
        .unwrap_or(&source);
    let needle = concat!("env::", "var");
    let reads = production
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .filter(|line| line.contains(needle))
        .count();
    assert!(
        reads <= 2,
        "config.rs reads {reads} environment variables in production code; the cap is 2"
    );
}
