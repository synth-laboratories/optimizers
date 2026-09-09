use serde_json::json;
use synth_optimizer_platform::cache::RequestCache;

#[test]
fn proposer_cache_reuses_evidence_across_run_delivery_locations() {
    let request = |root: &str, database_hash: &str| {
        json!({
            "model": "openai/gpt-4.1-mini",
            "parent": {"candidate_id": "seed", "prompt": "classify", "reward": 0.5},
            "workspace_root": root,
            "run_artifact_dir": root,
            "proposal_artifact_dir": format!("{root}/proposal"),
            "rollout_trace_artifact_refs": [{"kind": "rollout_trace_payload", "path": format!("{root}/trace.json"), "sha256": "trace-content"}],
            "merge_evidence_artifacts": [{"kind": "workspace_sqlite", "path": format!("{root}/workspace.sqlite"), "sha256": database_hash}]
        })
    };
    let key = |value: &serde_json::Value| {
        RequestCache::cache_key_with_profile("acceptance:proposer", value, "gepa_proposer")
    };
    let fresh = request("/fresh", "fresh-journal");
    let mut cached = request("/cached", "cache-hit-journal");
    assert_eq!(key(&fresh), key(&cached));
    cached["rollout_trace_artifact_refs"][0]["sha256"] = json!("changed-evidence");
    assert_ne!(key(&fresh), key(&cached));
    let mut changed = fresh.clone();
    changed["parent"]["prompt"] = json!("different prompt");
    assert_ne!(key(&fresh), key(&changed));
    changed = fresh.clone();
    changed["parent"]["reward"] = json!(1.0);
    assert_ne!(key(&fresh), key(&changed));
}
