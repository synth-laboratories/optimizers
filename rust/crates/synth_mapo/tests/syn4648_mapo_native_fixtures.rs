//! SYN-4648: authored MAPO TOML -> exact resolved native bytes, per MAPO config family.
//!
//! Fixtures live in `tests/fixtures/syn4648_mapo_rust/<family>/` at the repo root.
//! `MANIFEST.json` pins the sha256 of every input and of the frozen native bytes,
//! so a fixture cannot be silently rewritten alongside a resolver change.
//!
//! Families (both run the REAL parser/resolver, no re-implementation):
//!
//! * `mapo_config`   -- `MapoConfig::from_toml_file(path)?.resolved_config_value()`.
//! * `mapo_campaign` -- the same config bound to a frozen `debrief.campaign_manifest.v1`
//!   through `evidence.campaign_manifest_path`; native = resolved config plus
//!   `CampaignBinding::load(&config)` `evidence_value` and `receipt_value`.
//!
//! Native bytes = compact JSON with recursively sorted keys + trailing newline.
//! Absolute paths derived from the declaring document are written `${CONFIG_DIR}`.
//!
//! Regenerate deliberately (never in a normal test run):
//!     mbx test -p synth_mapo --test syn4648_mapo_native_fixtures -- --ignored regenerate

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use synth_mapo::{CampaignBinding, MapoConfig};
use synth_optimizer_platform::OptimizerError;

const CONFIG_DIR_TOKEN: &str = "${CONFIG_DIR}";
const MANIFEST_SCHEMA: &str = "syn4648.mapo-rust-fixtures.v1";
const FAMILIES: &[&str] = &["mapo_config", "mapo_campaign"];

fn fixtures_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../tests/fixtures/syn4648_mapo_rust")
        .canonicalize()
        .expect("syn4648_mapo_rust fixtures directory")
}

fn producer(family: &str) -> &'static str {
    match family {
        "mapo_config" => "synth_mapo::MapoConfig::from_toml_file().resolved_config_value()",
        "mapo_campaign" => {
            "synth_mapo::MapoConfig::from_toml_file() + CampaignBinding::load().{evidence_value,receipt_value}"
        }
        other => panic!("unknown family {other}"),
    }
}

/// Extra inputs (besides authored.toml) whose bytes the family depends on.
fn extra_inputs(family: &str) -> &'static [&'static str] {
    match family {
        "mapo_campaign" => &["campaign_manifest.json"],
        _ => &[],
    }
}

fn sort_value(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(String, Value)> = map.into_iter().collect();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            let mut sorted = Map::new();
            for (key, value) in entries {
                sorted.insert(key, sort_value(value));
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(sort_value).collect()),
        other => other,
    }
}

fn resolve(family: &str, authored: &Path) -> Value {
    let config = MapoConfig::from_toml_file(authored)
        .unwrap_or_else(|error| panic!("{family}: real resolver refused fixture: {error}"));
    let resolved = config
        .resolved_config_value()
        .expect("resolved config value");
    match family {
        "mapo_config" => resolved,
        "mapo_campaign" => {
            let binding = CampaignBinding::load(&config)
                .expect("campaign binding")
                .expect("campaign manifest path is set");
            json!({
                "resolved_config": resolved,
                "evidence": binding.evidence_value(&config),
                "receipt": binding.receipt_value(),
            })
        }
        other => panic!("unknown family {other}"),
    }
}

fn native_bytes(family: &str, authored: &Path) -> Vec<u8> {
    let value = sort_value(resolve(family, authored));
    let text = serde_json::to_string(&value).expect("serialize native value");
    let config_dir = authored.parent().expect("config dir").display().to_string();
    let text = text.replace(&config_dir, CONFIG_DIR_TOKEN);
    format!("{text}\n").into_bytes()
}

fn sha256(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn read(path: &Path) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn manifest() -> Value {
    serde_json::from_slice(&read(&fixtures_root().join("MANIFEST.json"))).expect("MANIFEST.json")
}

fn first_difference(expected: &[u8], actual: &[u8]) -> String {
    let index = expected
        .iter()
        .zip(actual)
        .position(|(left, right)| left != right)
        .unwrap_or_else(|| expected.len().min(actual.len()));
    let window = |bytes: &[u8]| {
        let start = index.saturating_sub(80);
        let end = (index + 80).min(bytes.len());
        String::from_utf8_lossy(&bytes[start.min(end)..end]).into_owned()
    };
    format!(
        "first difference at byte {index}\n  frozen: ...{}...\n  actual: ...{}...",
        window(expected),
        window(actual)
    )
}

#[test]
fn manifest_pins_every_family_input_and_native_bytes() {
    let manifest = manifest();
    assert_eq!(manifest["schema_version"], MANIFEST_SCHEMA);
    let families = manifest["families"].as_object().expect("families object");
    let mut names: Vec<&str> = families.keys().map(String::as_str).collect();
    names.sort_unstable();
    let mut expected: Vec<&str> = FAMILIES.to_vec();
    expected.sort_unstable();
    assert_eq!(names, expected, "MANIFEST families drifted");
    let root = fixtures_root();
    for family in FAMILIES {
        let entry = &families[*family];
        assert_eq!(entry["producer"], producer(family), "{family}: producer");
        let dir = root.join(family);
        assert_eq!(
            entry["authored_sha256"],
            sha256(&read(&dir.join("authored.toml"))),
            "{family}: authored.toml bytes differ from MANIFEST pin"
        );
        assert_eq!(
            entry["native_sha256"],
            sha256(&read(&dir.join("native.json"))),
            "{family}: native.json bytes differ from MANIFEST pin"
        );
        for input in extra_inputs(family) {
            assert_eq!(
                entry["inputs"][*input],
                sha256(&read(&dir.join(input))),
                "{family}: {input} bytes differ from MANIFEST pin"
            );
        }
    }
}

fn assert_family_matches_frozen_bytes(family: &str) {
    let dir = fixtures_root().join(family);
    let frozen = read(&dir.join("native.json"));
    let actual = native_bytes(family, &dir.join("authored.toml"));
    assert!(
        frozen == actual,
        "SYN-4648 {family}: real MAPO resolver output drifted from frozen native.json \
         (producer: {}).\n{}\nIf the change is intended, regenerate deliberately and review the diff.",
        producer(family),
        first_difference(&frozen, &actual)
    );
}

#[test]
fn mapo_config_resolves_to_frozen_native_bytes() {
    assert_family_matches_frozen_bytes("mapo_config");
}

#[test]
fn mapo_campaign_resolves_to_frozen_native_bytes() {
    assert_family_matches_frozen_bytes("mapo_campaign");
}

#[test]
fn native_bytes_are_compact_sorted_with_trailing_newline() {
    for family in FAMILIES {
        let bytes = read(&fixtures_root().join(family).join("native.json"));
        assert_eq!(bytes.last(), Some(&b'\n'), "{family}: trailing newline");
        assert_eq!(
            bytes.iter().filter(|byte| **byte == b'\n').count(),
            1,
            "{family}: compact (single line)"
        );
        let value: Value = serde_json::from_slice(&bytes).expect("native json");
        let canonical = format!("{}\n", serde_json::to_string(&sort_value(value)).unwrap());
        assert_eq!(
            canonical.as_bytes(),
            bytes.as_slice(),
            "{family}: canonical form"
        );
        assert!(
            !String::from_utf8_lossy(&bytes).contains(&fixtures_root().display().to_string()),
            "{family}: checkout path leaked into native bytes"
        );
    }
}

// ---------------------------------------------------------------- refusals

/// Write a mutated copy of a family's fixture into a fresh scratch dir under
/// the cargo target tmpdir and return the authored path.
fn scratch_fixture(name: &str, family: &str, edit: impl FnOnce(String) -> String) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("syn4648_mapo")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch dir");
    let source = fixtures_root().join(family);
    for input in extra_inputs(family) {
        fs::copy(source.join(input), dir.join(input)).expect("copy extra input");
    }
    let authored = String::from_utf8(read(&source.join("authored.toml"))).unwrap();
    let path = dir.join("authored.toml");
    fs::write(&path, edit(authored)).expect("write scratch authored.toml");
    path
}

fn mapo_error(path: &Path) -> OptimizerError {
    match MapoConfig::from_toml_file(path) {
        Ok(_) => panic!("{} was accepted; expected a refusal", path.display()),
        Err(error) => error,
    }
}

#[test]
fn mapo_config_refuses_unknown_top_level_key() {
    let path = scratch_fixture("config_unknown_top", "mapo_config", |text| {
        format!("{text}\n[surprise]\nenabled = true\n")
    });
    let error = mapo_error(&path);
    assert!(matches!(error, OptimizerError::TomlDecode(_)), "{error:?}");
    assert_eq!(error.error_code(), "synth_optimizer_toml_decode_error");
}

#[test]
fn mapo_config_refuses_unknown_nested_key() {
    let path = scratch_fixture("config_unknown_nested", "mapo_config", |text| {
        text.replace("max_steps = 64", "max_steps = 64\nmax_stepz = 65")
    });
    let error = mapo_error(&path);
    assert!(matches!(error, OptimizerError::TomlDecode(_)), "{error:?}");
    assert_eq!(error.error_code(), "synth_optimizer_toml_decode_error");
}

#[test]
fn mapo_config_refuses_version_declaration() {
    // MapoConfig has no schema/version field: a declared version (including the
    // newer `mapo.campaign.v2`) is an unknown key, not a silently ignored hint.
    let path = scratch_fixture("config_version", "mapo_config", |text| {
        format!("schema_version = \"mapo.campaign.v2\"\n{text}")
    });
    let error = mapo_error(&path);
    assert!(matches!(error, OptimizerError::TomlDecode(_)), "{error:?}");
    assert_eq!(error.error_code(), "synth_optimizer_toml_decode_error");
}

#[test]
fn mapo_config_refuses_unsupported_proposer_profile() {
    let path = scratch_fixture("config_profile", "mapo_config", |text| {
        text.replace(
            "proposer_mode = \"deterministic_grid\"",
            "proposer_mode = \"llm_freeform\"",
        )
    });
    let error = mapo_error(&path);
    assert!(matches!(error, OptimizerError::Config(_)), "{error:?}");
    assert_eq!(error.error_code(), "synth_optimizer_config_error");
}

#[test]
fn mapo_config_refuses_pool_path_segment_escape() {
    let path = scratch_fixture("config_pool_escape", "mapo_config", |text| {
        text.replace(
            "url = \"http://127.0.0.1:9921/\"",
            "url = \"http://127.0.0.1:9921/\"\npool = { pool_id = \"../escape\" }",
        )
    });
    let error = mapo_error(&path);
    assert!(matches!(error, OptimizerError::Config(_)), "{error:?}");
    assert_eq!(error.error_code(), "synth_optimizer_config_error");
}

#[test]
fn mapo_config_relative_paths_resolve_against_declaring_document() {
    let path = scratch_fixture("config_relative", "mapo_config", |text| text);
    let dir = path.parent().unwrap().to_path_buf();
    let config = MapoConfig::from_toml_file(&path).expect("fixture resolves");
    assert_eq!(config.run.output_dir, dir.join("runs"));
    assert_eq!(config.container.cwd, Some(dir.join("container")));
}

#[test]
fn mapo_campaign_refuses_unknown_manifest_field() {
    let path = scratch_fixture("campaign_unknown_field", "mapo_campaign", |text| text);
    let manifest_path = path.parent().unwrap().join("campaign_manifest.json");
    let mut manifest: Value = serde_json::from_slice(&read(&manifest_path)).unwrap();
    manifest["surprise"] = json!(true);
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let error = mapo_error(&path);
    assert!(matches!(error, OptimizerError::Json(_)), "{error:?}");
    assert_eq!(error.error_code(), "synth_optimizer_json_error");
}

#[test]
fn mapo_campaign_refuses_unknown_config_key() {
    let path = scratch_fixture("campaign_unknown_key", "mapo_campaign", |text| {
        text.replace(
            "claim_label = \"synth_measured\"",
            "claim_label = \"synth_measured\"\nclaim_labels = []",
        )
    });
    let error = mapo_error(&path);
    assert!(matches!(error, OptimizerError::TomlDecode(_)), "{error:?}");
    assert_eq!(error.error_code(), "synth_optimizer_toml_decode_error");
}

#[test]
fn mapo_campaign_refuses_unsupported_manifest_version() {
    let path = scratch_fixture("campaign_version", "mapo_campaign", |text| text);
    let manifest_path = path.parent().unwrap().join("campaign_manifest.json");
    let mut manifest: Value = serde_json::from_slice(&read(&manifest_path)).unwrap();
    manifest["schema_version"] = json!("mapo.campaign.v2");
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let error = mapo_error(&path);
    assert!(matches!(error, OptimizerError::Config(_)), "{error:?}");
    assert_eq!(error.error_code(), "synth_optimizer_config_error");
}

#[test]
fn mapo_campaign_refuses_unsupported_adapter_profile() {
    let path = scratch_fixture("campaign_adapter", "mapo_campaign", |text| text);
    let manifest_path = path.parent().unwrap().join("campaign_manifest.json");
    let mut manifest: Value = serde_json::from_slice(&read(&manifest_path)).unwrap();
    manifest["adapter_id"] = json!("mapo-unknown-bench");
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let error = mapo_error(&path);
    assert!(matches!(error, OptimizerError::Json(_)), "{error:?}");
    assert_eq!(error.error_code(), "synth_optimizer_json_error");
}

#[test]
fn mapo_campaign_manifest_path_is_config_relative_not_cwd_relative() {
    // Copy only authored.toml: the relative manifest path must be looked up next
    // to the declaring document and fail there with a typed Io error.
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("syn4648_mapo")
        .join("campaign_relative_missing");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("authored.toml");
    fs::copy(fixtures_root().join("mapo_campaign/authored.toml"), &path).unwrap();
    let error = mapo_error(&path);
    match &error {
        OptimizerError::Io { path: missing, .. } => {
            assert_eq!(missing, &dir.join("campaign_manifest.json"))
        }
        other => panic!("expected Io for config-relative manifest, got {other:?}"),
    }
    assert_eq!(error.error_code(), "synth_optimizer_io_error");
}

// ------------------------------------------------------------- regenerate

#[test]
#[ignore = "regenerates frozen fixtures; run deliberately with --ignored regenerate"]
fn regenerate() {
    let root = fixtures_root();
    let mut families = Map::new();
    for family in FAMILIES {
        let dir = root.join(family);
        let native = native_bytes(family, &dir.join("authored.toml"));
        fs::write(dir.join("native.json"), &native).unwrap();
        let mut entry = Map::new();
        entry.insert("producer".into(), json!(producer(family)));
        entry.insert(
            "authored_sha256".into(),
            json!(sha256(&read(&dir.join("authored.toml")))),
        );
        entry.insert("native_sha256".into(), json!(sha256(&native)));
        if !extra_inputs(family).is_empty() {
            let mut inputs = Map::new();
            for input in extra_inputs(family) {
                inputs.insert((*input).into(), json!(sha256(&read(&dir.join(input)))));
            }
            entry.insert("inputs".into(), Value::Object(inputs));
        }
        families.insert((*family).into(), Value::Object(entry));
    }
    let manifest = sort_value(json!({"schema_version": MANIFEST_SCHEMA, "families": families}));
    let text = serde_json::to_string_pretty(&manifest).unwrap();
    fs::write(root.join("MANIFEST.json"), format!("{text}\n")).unwrap();
}
