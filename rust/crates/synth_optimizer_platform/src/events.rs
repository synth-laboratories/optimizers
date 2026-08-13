use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use time::OffsetDateTime;

use crate::cache::normalize_for_cache;
use crate::disk_budget::DiskBudget;
use crate::error::{OptimizerError, Result};
use crate::event_visualization::{
    render_terminal_event, terminal_events_enabled, terminal_line_for_event,
};
use crate::observability::{
    proposer_delta_fields, OptimizerAlgorithm, OptimizerEvent, CHILD_ROLLOUT_ATTACHED_EVENT_TYPE,
    PROPOSER_DELTA_EVENT_TYPE,
};

/// Sibling canonical feed path: `events.jsonl` → `events.optimizer.jsonl`.
pub fn optimizer_event_feed_path_for(event_feed: impl AsRef<Path>) -> PathBuf {
    let event_feed = event_feed.as_ref();
    let file_name = event_feed
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("events.jsonl");
    let stem = file_name.strip_suffix(".jsonl").unwrap_or(file_name);
    event_feed.with_file_name(format!("{stem}.optimizer.jsonl"))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EventStreamRecord {
    pub schema_version: String,
    pub event_id: String,
    pub sequence_number: u64,
    pub event_type: String,
    pub message: String,
    pub timestamp: String,
    pub fields: Value,
    pub event: Value,
}

pub struct EventWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    optimizer_path: PathBuf,
    optimizer_writer: BufWriter<File>,
    records: Vec<EventStreamRecord>,
    disk_budget: Option<DiskBudget>,
    run_id: String,
    algorithm: OptimizerAlgorithm,
}

impl EventWriter {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        Self::open(path, false, "unknown", OptimizerAlgorithm::Gepa)
    }

    pub fn append(path: impl AsRef<Path>) -> Result<Self> {
        Self::open(path, true, "unknown", OptimizerAlgorithm::Gepa)
    }

    fn open(
        path: impl AsRef<Path>,
        resume: bool,
        run_id: &str,
        algorithm: OptimizerAlgorithm,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| OptimizerError::io(parent, source))?;
        }
        let optimizer_path = optimizer_event_feed_path_for(&path);
        let records = if resume && path.exists() {
            read_existing_records(&path)?
        } else {
            Vec::new()
        };
        let file = if resume {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|source| OptimizerError::io(&path, source))?
        } else {
            File::create(&path).map_err(|source| OptimizerError::io(&path, source))?
        };
        let optimizer_file = if resume {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&optimizer_path)
                .map_err(|source| OptimizerError::io(&optimizer_path, source))?
        } else {
            File::create(&optimizer_path)
                .map_err(|source| OptimizerError::io(&optimizer_path, source))?
        };
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            optimizer_path,
            optimizer_writer: BufWriter::new(optimizer_file),
            records,
            disk_budget: None,
            run_id: run_id.to_string(),
            algorithm,
        })
    }

    /// Attach run identity used when dual-writing `optimizer_event.v1`.
    pub fn with_optimizer_context(
        mut self,
        run_id: impl Into<String>,
        algorithm: OptimizerAlgorithm,
    ) -> Self {
        self.run_id = run_id.into();
        self.algorithm = algorithm;
        self
    }

    /// Attach a [`DiskBudget`] so each emit checks the hard limit before
    /// touching the file. Returns `self` for builder-style chaining at
    /// the construction site.
    pub fn with_disk_budget(mut self, disk_budget: DiskBudget) -> Self {
        self.disk_budget = Some(disk_budget);
        self
    }

    pub fn emit(&mut self, event_type: &str, message: &str, fields: Value) -> Result<()> {
        // Hard-limit gate: refuse the write before we corrupt the jsonl
        // by partial-appending under ENOSPC. Soft-limit is enforced at
        // run-start, not here.
        if let Some(budget) = &self.disk_budget {
            budget.require_below_hard()?;
        }
        let timestamp = OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string());
        let event = serde_json::json!({
            "ts": timestamp.clone(),
            "type": event_type,
            "message": message,
            "fields": fields.clone(),
        });
        let line = serde_json::to_string(&event)?;
        let bytes_written = (line.len() + 1) as u64; // +1 for the newline
        writeln!(self.writer, "{line}").map_err(|source| OptimizerError::io(&self.path, source))?;
        self.writer
            .flush()
            .map_err(|source| OptimizerError::io(&self.path, source))?;
        self.writer
            .get_ref()
            .sync_data()
            .map_err(|source| OptimizerError::io(&self.path, source))?;

        let sequence_number = self.records.len() as u64 + 1;
        let run_id = fields
            .get("run_id")
            .and_then(Value::as_str)
            .unwrap_or(&self.run_id);
        let canonical = OptimizerEvent::from_gepa_stream(
            sequence_number,
            event_type,
            message,
            timestamp.clone(),
            fields.clone(),
            run_id,
            self.algorithm,
        );
        let canonical_line = serde_json::to_string(&canonical)?;
        let canonical_bytes = (canonical_line.len() + 1) as u64;
        writeln!(self.optimizer_writer, "{canonical_line}")
            .map_err(|source| OptimizerError::io(&self.optimizer_path, source))?;
        self.optimizer_writer
            .flush()
            .map_err(|source| OptimizerError::io(&self.optimizer_path, source))?;
        // The canonical optimizer stream is a publication boundary. A poll or
        // WebSocket consumer must never observe an event that only lives in a
        // userspace buffer or the kernel page cache.
        self.optimizer_writer
            .get_ref()
            .sync_data()
            .map_err(|source| OptimizerError::io(&self.optimizer_path, source))?;

        if let Some(budget) = &self.disk_budget {
            budget.note_appended_bytes(bytes_written.saturating_add(canonical_bytes));
        }
        if terminal_events_enabled() {
            render_terminal_event(event_type, message, &fields);
        }
        self.records.push(EventStreamRecord::new(
            sequence_number,
            event_type,
            message,
            timestamp,
            event,
        ));
        Ok(())
    }

    /// Append a live proposer text chunk onto this run's `optimizer_event.v1` spool.
    /// Empty text is dropped. This is not a second stream.
    pub fn emit_proposer_delta(
        &mut self,
        generation: u64,
        channel: &str,
        text: &str,
    ) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        self.emit(
            PROPOSER_DELTA_EVENT_TYPE,
            "",
            Value::Object(proposer_delta_fields(generation, channel, text)),
        )
    }

    /// Link a Containers child eval. Frames and NEV kinds stay on the eval stream.
    pub fn emit_child_rollout_attached(&mut self, child_resource_ref: Value) -> Result<()> {
        self.emit(
            CHILD_ROLLOUT_ATTACHED_EVENT_TYPE,
            "Child rollout resource reference attached",
            serde_json::json!({"child_resource_ref": child_resource_ref}),
        )
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer
            .flush()
            .map_err(|source| OptimizerError::io(&self.path, source))?;
        self.writer
            .get_ref()
            .sync_data()
            .map_err(|source| OptimizerError::io(&self.path, source))?;
        self.optimizer_writer
            .flush()
            .map_err(|source| OptimizerError::io(&self.optimizer_path, source))?;
        self.optimizer_writer
            .get_ref()
            .sync_data()
            .map_err(|source| OptimizerError::io(&self.optimizer_path, source))
    }

    pub fn records(&self) -> &[EventStreamRecord] {
        &self.records
    }

    pub fn optimizer_event_feed_path(&self) -> &Path {
        &self.optimizer_path
    }
}

fn read_existing_records(path: &Path) -> Result<Vec<EventStreamRecord>> {
    let file = File::open(path).map_err(|source| OptimizerError::io(path, source))?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|source| OptimizerError::io(path, source))?;
        if line.trim().is_empty() {
            continue;
        }
        let event = serde_json::from_str::<Value>(&line)?;
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("event")
            .to_string();
        let message = event
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let timestamp = event
            .get("ts")
            .and_then(Value::as_str)
            .unwrap_or("1970-01-01T00:00:00Z")
            .to_string();
        records.push(EventStreamRecord::new(
            records.len() as u64 + 1,
            &event_type,
            &message,
            timestamp,
            event,
        ));
    }
    Ok(records)
}

impl EventStreamRecord {
    fn new(
        sequence_number: u64,
        event_type: &str,
        message: &str,
        timestamp: String,
        event: Value,
    ) -> Self {
        let fields = event.get("fields").cloned().unwrap_or(Value::Null);
        Self {
            schema_version: "event_stream_record.v1".to_string(),
            event_id: stable_id(
                "event",
                &[&sequence_number.to_string(), event_type, message],
            ),
            sequence_number,
            event_type: event_type.to_string(),
            message: message.to_string(),
            timestamp,
            fields,
            event,
        }
    }
}

pub fn replay_event_feed(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|source| OptimizerError::io(path, source))?;
    let reader = BufReader::new(file);
    let mut out = String::new();
    for line in reader.lines() {
        let line = line.map_err(|source| OptimizerError::io(path, source))?;
        if line.trim().is_empty() {
            continue;
        }
        let value = serde_json::from_str::<Value>(&line)?;
        let event_type = value.get("type").and_then(Value::as_str).unwrap_or("event");
        let message = value.get("message").and_then(Value::as_str).unwrap_or("");
        let fields = value.get("fields").unwrap_or(&Value::Null);
        if let Some(line) = terminal_line_for_event(event_type, message, fields) {
            out.push_str(&line);
            out.push('\n');
        }
    }
    Ok(out)
}

pub fn normalize_event_feed(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    artifact_root: impl AsRef<Path>,
) -> Result<()> {
    let input = input.as_ref();
    let output = output.as_ref();
    let artifact_root = artifact_root.as_ref().display().to_string();
    let file = File::open(input).map_err(|source| OptimizerError::io(input, source))?;
    let reader = BufReader::new(file);
    let mut lines = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|source| OptimizerError::io(input, source))?;
        if line.trim().is_empty() {
            continue;
        }
        let mut value = serde_json::from_str::<Value>(&line)?;
        value = normalize_event_value(value, &artifact_root);
        lines.push(serde_json::to_string(&value)?);
    }
    fs::write(output, format!("{}\n", lines.join("\n")))
        .map_err(|source| OptimizerError::io(output, source))
}

pub fn compare_normalized_event_feeds(
    left: impl AsRef<Path>,
    right: impl AsRef<Path>,
) -> Result<()> {
    let left = left.as_ref();
    let right = right.as_ref();
    let left_text = fs::read_to_string(left).map_err(|source| OptimizerError::io(left, source))?;
    let right_text =
        fs::read_to_string(right).map_err(|source| OptimizerError::io(right, source))?;
    if left_text == right_text {
        return Ok(());
    }
    Err(OptimizerError::EventCompare(format!(
        "{} differs from {}",
        left.display(),
        right.display()
    )))
}

fn normalize_event_value(value: Value, artifact_root: &str) -> Value {
    match value {
        Value::Object(map) => {
            let mut out = Map::new();
            for (key, item) in map {
                if key == "ts" || key == "at" {
                    continue;
                }
                out.insert(key, normalize_event_value(item, artifact_root));
            }
            normalize_for_cache(&Value::Object(out))
        }
        Value::Array(items) => Value::Array(
            items
                .into_iter()
                .map(|item| normalize_event_value(item, artifact_root))
                .collect(),
        ),
        Value::String(text) => Value::String(text.replace(artifact_root, "{ARTIFACT_ROOT}")),
        _ => value,
    }
}

fn stable_id(prefix: &str, parts: &[&str]) -> String {
    let mut digest = Sha256::new();
    digest.update(prefix.as_bytes());
    for part in parts {
        digest.update(b"\0");
        digest.update(part.as_bytes());
    }
    let hex = format!("{:x}", digest.finalize());
    format!("{prefix}_{}", &hex[..16])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::{
        CHILD_ROLLOUT_ATTACHED_EVENT_TYPE, DEFAULT_PROPOSER_DELTA_CHANNEL, PROPOSER_DELTA_EVENT_TYPE,
    };
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("synth_opt_events_{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn dual_writes_optimizer_event_sidecar_for_gepa() {
        let dir = scratch_dir();
        let path = dir.join("events.jsonl");
        let mut writer = EventWriter::new(&path)
            .unwrap()
            .with_optimizer_context("gepa_test_1", OptimizerAlgorithm::Gepa);
        writer
            .emit(
                "gepa.run.started",
                "started",
                json!({"run_id": "gepa_test_1"}),
            )
            .unwrap();
        writer
            .emit(
                "candidate.accepted",
                "accepted",
                json!({"run_id": "gepa_test_1", "candidate_id": "c1", "train_reward": 0.5}),
            )
            .unwrap();

        let legacy = fs::read_to_string(&path).unwrap();
        assert!(legacy.contains("\"type\":\"gepa.run.started\""));
        assert!(!legacy.contains("synth.stream-event.v1"));

        let canonical = fs::read_to_string(writer.optimizer_event_feed_path()).unwrap();
        let lines: Vec<&str> = canonical
            .lines()
            .filter(|line| !line.trim().is_empty())
            .collect();
        assert_eq!(lines.len(), 2);
        let second: Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(second["schema_version"], "optimizer_event.v1");
        assert_eq!(second["sequence_number"], 2);
        assert_eq!(second["run_id"], "gepa_test_1");
        assert_eq!(second["algorithm_id"], "gepa");
        assert_eq!(second["slot"], "optimizer_run");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_spools_do_not_cross_logs() {
        let dir = scratch_dir();
        let mut luna = EventWriter::new(dir.join("luna").join("events.jsonl"))
            .unwrap()
            .with_optimizer_context("gepa_luna", OptimizerAlgorithm::Gepa);
        let mut sol = EventWriter::new(dir.join("sol").join("events.jsonl"))
            .unwrap()
            .with_optimizer_context("gepa_sol", OptimizerAlgorithm::Gepa);
        luna.emit(
            "optimizer.run.started",
            "started",
            json!({"run_id": "gepa_luna"}),
        )
        .unwrap();
        sol.emit(
            "optimizer.run.started",
            "started",
            json!({"run_id": "gepa_sol"}),
        )
        .unwrap();
        luna.emit(
            "optimizer.candidate.added",
            "added",
            json!({"run_id": "gepa_luna"}),
        )
        .unwrap();

        let luna_lines: Vec<Value> = fs::read_to_string(luna.optimizer_event_feed_path())
            .unwrap()
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let sol_lines: Vec<Value> = fs::read_to_string(sol.optimizer_event_feed_path())
            .unwrap()
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(luna_lines.len(), 2);
        assert_eq!(sol_lines.len(), 1);
        assert_eq!(luna_lines[0]["run_id"], "gepa_luna");
        assert_eq!(sol_lines[0]["run_id"], "gepa_sol");
        assert_eq!(luna_lines[1]["sequence_number"], 2);
        assert_eq!(sol_lines[0]["sequence_number"], 1);
        assert_ne!(
            luna.optimizer_event_feed_path(),
            sol.optimizer_event_feed_path()
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn proposer_delta_dual_writes_on_the_same_optimizer_stream() {
        let dir = scratch_dir();
        let path = dir.join("events.jsonl");
        let mut writer = EventWriter::new(&path)
            .unwrap()
            .with_optimizer_context("gepa_luna", OptimizerAlgorithm::Gepa);
        writer
            .emit(
                "optimizer.run.started",
                "started",
                json!({"run_id": "gepa_luna"}),
            )
            .unwrap();
        writer
            .emit_proposer_delta(0, "reasoning", "chunk0 ")
            .unwrap();
        writer
            .emit_proposer_delta(0, DEFAULT_PROPOSER_DELTA_CHANNEL, "Final proposal.")
            .unwrap();
        writer.emit_proposer_delta(0, "content", "").unwrap();
        writer
            .emit_child_rollout_attached(crate::container_child_eval_ref(
                "rollout_abc",
                "stream:abc",
                "/reward?rollout_id=rollout_abc",
            ))
            .unwrap();

        let canonical = fs::read_to_string(writer.optimizer_event_feed_path()).unwrap();
        let lines: Vec<Value> = canonical
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 4);
        assert_eq!(lines[1]["type"], PROPOSER_DELTA_EVENT_TYPE);
        assert_eq!(lines[1]["sequence_number"], 2);
        assert_eq!(lines[1]["delta"]["generation"], 0);
        assert_eq!(lines[1]["delta"]["channel"], "reasoning");
        assert_eq!(lines[1]["delta"]["text"], "chunk0 ");
        assert!(lines[1]["delta"].get("message").is_none());
        assert_eq!(lines[2]["delta"]["channel"], "content");
        assert_eq!(lines[2]["delta"]["text"], "Final proposal.");
        assert_eq!(lines[3]["type"], CHILD_ROLLOUT_ATTACHED_EVENT_TYPE);
        assert_eq!(lines[3]["sequence_number"], 4);
        let child = &lines[3]["delta"]["child_resource_ref"];
        assert_eq!(child["schema"], "synth.resource-ref.v1");
        assert_eq!(child["kind"], "container_rollout");
        assert_eq!(child["id"], "rollout_abc");
        assert_eq!(child["attributes"]["stream_id"], "stream:abc");
        assert_eq!(
            child["attributes"]["reward_url"],
            "/reward?rollout_id=rollout_abc"
        );
        assert!(!canonical.to_ascii_lowercase().contains("nev"));
        assert!(!canonical.contains("child_eval_ref"));
        assert!(!canonical.contains("synth.trace-stream-event.v1"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn optimizer_event_feed_path_for_siblings_events_jsonl() {
        let path = PathBuf::from("/tmp/run/events.jsonl");
        assert_eq!(
            optimizer_event_feed_path_for(&path),
            PathBuf::from("/tmp/run/events.optimizer.jsonl")
        );
    }
}
