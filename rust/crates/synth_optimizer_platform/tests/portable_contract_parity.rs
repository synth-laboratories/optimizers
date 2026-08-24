use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use synth_optimizer_platform::portable_contracts::{canonical_digest, canonical_json};

#[test]
fn rust_matches_the_shared_golden_corpus() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../contracts/synth-spine-v1/fixtures/valid/canonical-values.json");
    let corpus: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    for case in corpus["cases"].as_array().unwrap() {
        let omit = case
            .get("omit_self_digest")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        assert_eq!(
            String::from_utf8(canonical_json(&case["value"], omit).unwrap()).unwrap(),
            case["canonical"]
        );
        assert_eq!(
            canonical_digest(&case["value"], omit).unwrap(),
            case["digest"]
        );
    }
}

#[test]
fn floats_and_out_of_range_integers_fail_closed() {
    let float = serde_json::json!({"n": 1.5});
    assert_eq!(
        canonical_digest(&float, false).unwrap_err().code,
        "canonical_float_forbidden"
    );
    let large = serde_json::json!({"n": 9_007_199_254_740_992_u64});
    assert_eq!(
        canonical_digest(&large, false).unwrap_err().code,
        "canonical_integer_range"
    );
}
