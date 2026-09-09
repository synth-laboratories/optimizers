use super::*;
use std::sync::{Arc, Barrier};

#[test]
fn concurrent_appends_remain_distinct_valid_jsonl_records() {
    let home = std::env::temp_dir().join(format!(
        "synth_gepa_index_concurrency_{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let barrier = Arc::new(Barrier::new(3));
    let mut writers = Vec::new();
    for run_id in ["gepa_luna", "gepa_sol"] {
        let home = home.clone();
        let barrier = Arc::clone(&barrier);
        writers.push(thread::spawn(move || {
            let entry = json!({
                "schema": "synth.gepa_run_index.v1",
                "run_id": run_id,
                "run_dir": home.join(run_id),
                "event_feed_path": home.join(run_id).join("optimizer_events.jsonl"),
            });
            barrier.wait();
            append_global_gepa_run_index_entry(&home, &entry).unwrap();
        }));
    }
    barrier.wait();
    for writer in writers {
        writer.join().unwrap();
    }

    let lines = fs::read_to_string(home.join("index.jsonl")).unwrap();
    let entries = lines
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(entries.len(), 2);
    assert_eq!(
        entries
            .iter()
            .filter_map(|entry| entry.get("run_id").and_then(Value::as_str))
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["gepa_luna", "gepa_sol"])
    );
    fs::remove_dir_all(home).unwrap();
}
