//! SYN-4648: Rust GEPA parity with the authored GEPA TOML the Python fixture test
//! freezes (`tests/fixtures/syn4648_cli_toml/gepa/authored.toml`).
//!
//! The Rust GEPA config path is `SynthOptimizerConfig::from_toml_file` (used by
//! `synth_gepa`), which embeds the shared-platform `ContainerConfig`. These tests
//! parse the SAME authored bytes and pin what Rust does with them.

use std::fs;
use std::path::{Path, PathBuf};

use synth_optimizer_platform::{OptimizerError, SynthOptimizerConfig};

fn gepa_fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../tests/fixtures/syn4648_cli_toml/gepa/authored.toml")
        .canonicalize()
        .expect("shared GEPA authored.toml fixture")
}

fn authored_text() -> String {
    fs::read_to_string(gepa_fixture()).expect("read GEPA authored.toml")
}

/// Write a mutated copy next to a scratch `prompts/` dir so config-relative
/// paths still resolve the way they do for the checked-in fixture.
fn scratch(name: &str, text: String) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("syn4648_gepa")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("authored.toml");
    fs::write(&path, text).unwrap();
    path
}

const AUTH_REFRESH: &str = r#"
[container.auth_refresh]
provider = "synth_tunnel_v2"
lease_id = "lease_syn4648"
connector_id = "connector_syn4648"
refresh_interval_seconds = 240
"#;

/// Insert a `[container.auth_refresh]` table right after the `[container]` table.
fn with_auth_refresh(text: &str, table: &str) -> String {
    text.replacen("\n[taskset]", &format!("{table}\n[taskset]"), 1)
}

#[test]
fn rust_gepa_accepts_shared_authored_fixture() {
    let config = SynthOptimizerConfig::from_toml_file(gepa_fixture())
        .unwrap_or_else(|error| panic!("Rust refused the shared GEPA fixture: {error:?}"));
    let dir = gepa_fixture().parent().unwrap().to_path_buf();
    assert_eq!(config.run.run_id, "syn4648_gepa_fixture");
    assert_eq!(config.run.output_dir, dir.join("runs"));
    assert_eq!(
        config.container.url.as_deref(),
        Some("http://127.0.0.1:9911/")
    );
    assert_eq!(config.container.cwd, Some(dir.join("container")));
    assert_eq!(config.container.startup_timeout_seconds, 45);
    assert!(config.container.auth_refresh.is_none());
    assert_eq!(config.taskset.train_ids, ["train:0", "train:1"]);
    assert_eq!(config.gepa.task_pools.minibatch, ["train:0"]);
    assert_eq!(
        config.proposer.prompt.best_practices_path,
        Some(dir.join("prompts/best_practices.md"))
    );
    // Divergence pinned, not endorsed: Rust leaves proposer.codex_home exactly as
    // authored, while Python's GEPA native resolves it to ${CONFIG_DIR}/codex-home
    // and leaves container.cwd / run.output_dir relative (the reverse of Rust).
    assert_eq!(
        config.proposer.codex_home,
        Some(PathBuf::from("codex-home"))
    );
}

#[test]
fn rust_gepa_refuses_unknown_top_level_key() {
    let path = scratch(
        "unknown_top_level",
        format!("{}\n[surprise]\nenabled = true\n", authored_text()),
    );
    let error = SynthOptimizerConfig::from_toml_file(&path).expect_err("unknown key accepted");
    assert!(matches!(error, OptimizerError::TomlDecode(_)), "{error:?}");
    assert_eq!(error.error_code(), "synth_optimizer_toml_decode_error");
}

#[test]
fn rust_gepa_refuses_unknown_container_key() {
    let path = scratch(
        "unknown_container_key",
        authored_text().replacen(
            "startup_timeout_seconds = 45",
            "startup_timeout_seconds = 45\nstartup_timeout = 45",
            1,
        ),
    );
    let error = SynthOptimizerConfig::from_toml_file(&path).expect_err("unknown key accepted");
    assert!(matches!(error, OptimizerError::TomlDecode(_)), "{error:?}");
}

#[test]
fn rust_gepa_preserves_container_auth_refresh() {
    let path = scratch(
        "auth_refresh",
        with_auth_refresh(&authored_text(), AUTH_REFRESH),
    );
    let config = SynthOptimizerConfig::from_toml_file(&path).expect("auth_refresh accepted");
    let refresh = config
        .container
        .auth_refresh
        .as_ref()
        .expect("auth_refresh kept");
    assert_eq!(refresh.provider, "synth_tunnel_v2");
    assert_eq!(refresh.lease_id, "lease_syn4648");
    assert_eq!(refresh.connector_id.as_deref(), Some("connector_syn4648"));
    assert_eq!(refresh.refresh_interval_seconds, Some(240));
    assert_eq!(refresh.offer_id, None);
    assert_eq!(refresh.job_id, None);
    // Round-trips through the serialized config handed to the GEPA runtime.
    let value = serde_json::to_value(&config).unwrap();
    assert_eq!(
        value["container"]["auth_refresh"],
        serde_json::json!({
            "provider": "synth_tunnel_v2",
            "lease_id": "lease_syn4648",
            "connector_id": "connector_syn4648",
            "refresh_interval_seconds": 240,
        })
    );
}

#[test]
fn rust_gepa_refuses_unknown_auth_refresh_key() {
    let table = format!("{AUTH_REFRESH}lease_ttl_seconds = 60\n");
    let path = scratch(
        "auth_refresh_unknown",
        with_auth_refresh(&authored_text(), &table),
    );
    let error = SynthOptimizerConfig::from_toml_file(&path).expect_err("unknown key accepted");
    assert!(matches!(error, OptimizerError::TomlDecode(_)), "{error:?}");
}

#[test]
fn rust_gepa_refuses_auth_refresh_without_lease_id() {
    let table = AUTH_REFRESH.replace("lease_id = \"lease_syn4648\"\n", "");
    let path = scratch(
        "auth_refresh_no_lease",
        with_auth_refresh(&authored_text(), &table),
    );
    let error = SynthOptimizerConfig::from_toml_file(&path).expect_err("missing lease accepted");
    assert!(matches!(error, OptimizerError::TomlDecode(_)), "{error:?}");
}

#[test]
#[ignore = "SYN-4648 gap: Rust ContainerAuthRefreshConfig.provider is a free String with no \
            validation and no runtime consumer of container.auth_refresh exists in rust/crates; \
            an unknown provider is accepted silently"]
fn rust_gepa_refuses_unknown_auth_refresh_provider() {
    let table = AUTH_REFRESH.replace("synth_tunnel_v2", "not_a_provider");
    let path = scratch(
        "auth_refresh_bad_provider",
        with_auth_refresh(&authored_text(), &table),
    );
    let error = SynthOptimizerConfig::from_toml_file(&path).expect_err("unknown provider accepted");
    assert!(matches!(error, OptimizerError::Config(_)), "{error:?}");
}
