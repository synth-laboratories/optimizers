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
    records: Vec<EventStreamRecord>,
    disk_budget: Option<DiskBudget>,
}

impl EventWriter {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| OptimizerError::io(parent, source))?;
        }
        let file = File::create(&path).map_err(|source| OptimizerError::io(&path, source))?;
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            records: Vec::new(),
            disk_budget: None,
        })
    }

    pub fn append(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| OptimizerError::io(parent, source))?;
        }
        let records = if path.exists() {
            read_existing_records(&path)?
        } else {
            Vec::new()
        };
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| OptimizerError::io(&path, source))?;
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            records,
            disk_budget: None,
        })
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
        if let Some(budget) = &self.disk_budget {
            budget.note_appended_bytes(bytes_written);
        }
        if terminal_events_enabled() {
            render_terminal_event(event_type, message, &fields);
        }
        self.records.push(EventStreamRecord::new(
            self.records.len() as u64 + 1,
            event_type,
            message,
            timestamp,
            event,
        ));
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer
            .flush()
            .map_err(|source| OptimizerError::io(&self.path, source))
    }

    pub fn records(&self) -> &[EventStreamRecord] {
        &self.records
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
