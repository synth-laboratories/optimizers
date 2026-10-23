//! Provider-free protocol probe; point only at a local fixture producer.
use serde_json::json;
use synth_optimizer_platform::http::ContainerClient;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let url = std::env::args().nth(1).ok_or("local fixture URL required")?;
    if !url.starts_with("http://127.0.0.1:") {
        return Err("probe requires loopback fixture".into());
    }
    let client = ContainerClient::new(url)?;
    let metadata = client.metadata()?;
    assert_eq!(metadata["policy_revision"]["schema_version"], "synth.policy-revision.v1");
    let mut revisions = Vec::new();
    for (index, prompt) in ["Policy A", "Policy B", "Policy A"].iter().enumerate() {
        let result = client.rollout(&json!({
            "rollout_id": format!("conformance_{index}"), "submission_mode": "sync",
            "candidate": {"stage2_system": prompt},
            "candidate_overlay": {"candidate": {"stage2_system": prompt}},
            "policy": {"provider": "fixture", "model": "fixture"},
            "task": {"task_id": "train:0", "text": "Card delivery", "label": "card_arrival"}
        }))?;
        assert_eq!(result["status"], "completed");
        let revision = result["policy_revision_id"].as_str().ok_or("missing binding")?.to_string();
        let readback = client.rollout_record(&format!("conformance_{index}"))?;
        assert_eq!(readback["policy_revision_id"], revision);
        revisions.push(revision);
    }
    assert_eq!(revisions[0], revisions[2]);
    assert_ne!(revisions[0], revisions[1]);
    println!("{}", json!({"schema": "policy-revision.conformance.v1", "revisions": revisions}));
    Ok(())
}
