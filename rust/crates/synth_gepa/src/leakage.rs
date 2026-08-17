use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const DEFAULT_LEAKAGE_MIN_SPAN_CHARS: usize = 32;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LeakageMatch {
    pub example_id: String,
    pub span: String,
    pub chars: usize,
}

#[derive(Clone, Debug)]
pub struct LeakageExample<'a> {
    pub example_id: &'a str,
    pub text: &'a str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeakagePolicy {
    Forbid,
    Warn,
    Allow,
}

impl LeakagePolicy {
    pub fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "allow" | "off" | "disabled" => Self::Allow,
            "warn" | "warning" => Self::Warn,
            _ => Self::Forbid,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Forbid => "forbid",
            Self::Warn => "warn",
            Self::Allow => "allow",
        }
    }

    /// Run-config `forbid` beats a producer `allow` hint.
    pub fn effective(self, producer_hint: Option<&str>) -> Self {
        if matches!(self, Self::Forbid) {
            return Self::Forbid;
        }
        if let Some(hint) = producer_hint {
            let hint = hint.to_ascii_lowercase();
            if hint.contains("forbid") || hint.contains("disallow") {
                return Self::Forbid;
            }
        }
        self
    }
}

pub fn scan_text_for_leakage(
    candidate_text: &str,
    examples: &[LeakageExample<'_>],
    min_span_chars: usize,
) -> Option<LeakageMatch> {
    let haystack = candidate_text.to_ascii_lowercase();
    if haystack.trim().is_empty() {
        return None;
    }
    for example in examples {
        let needle = example.text.trim().to_ascii_lowercase();
        if needle.len() < min_span_chars {
            continue;
        }
        if haystack.contains(&needle) {
            return Some(LeakageMatch {
                example_id: example.example_id.to_string(),
                span: example.text.trim().to_string(),
                chars: needle.chars().count(),
            });
        }
    }
    None
}

pub fn leakage_examples_from_rows(rows: &[Value]) -> Vec<(String, String)> {
    rows.iter()
        .filter_map(|row| {
            let example_id = string_field(row, &["example_id", "id", "task_id"])?;
            let text = string_field(
                row,
                &[
                    "text",
                    "query",
                    "input",
                    "prompt",
                    "utterance",
                    "question",
                    "content",
                ],
            )?;
            Some((example_id, text))
        })
        .collect()
}

pub fn payload_text(payload: &std::collections::BTreeMap<String, String>) -> String {
    payload
        .values()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn scan_payload_for_leakage(
    payload: &std::collections::BTreeMap<String, String>,
    rows: &[Value],
    min_span_chars: usize,
) -> Option<LeakageMatch> {
    let owned = leakage_examples_from_rows(rows);
    let examples = owned
        .iter()
        .map(|(example_id, text)| LeakageExample { example_id, text })
        .collect::<Vec<_>>();
    scan_text_for_leakage(&payload_text(payload), &examples, min_span_chars)
}

fn string_field(row: &Value, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(value) = row.get(*key).and_then(Value::as_str) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn thirty_two_char_containment_is_a_leak() {
        let example = "this training sentence is long enough to trip";
        assert!(example.len() >= DEFAULT_LEAKAGE_MIN_SPAN_CHARS);
        let hit = scan_text_for_leakage(
            &format!("prefix {example} suffix"),
            &[LeakageExample {
                example_id: "ex-1",
                text: example,
            }],
            DEFAULT_LEAKAGE_MIN_SPAN_CHARS,
        )
        .expect("leak");
        assert_eq!(hit.example_id, "ex-1");
        assert_eq!(hit.chars, example.chars().count());
    }

    #[test]
    fn short_spans_are_ignored() {
        assert!(scan_text_for_leakage(
            "hello world",
            &[LeakageExample {
                example_id: "ex-1",
                text: "hello",
            }],
            DEFAULT_LEAKAGE_MIN_SPAN_CHARS,
        )
        .is_none());
    }

    #[test]
    fn run_forbid_beats_producer_allow() {
        assert_eq!(
            LeakagePolicy::Forbid.effective(Some("allow")),
            LeakagePolicy::Forbid
        );
        assert_eq!(
            LeakagePolicy::Allow.effective(Some("forbid")),
            LeakagePolicy::Forbid
        );
        assert_eq!(
            LeakagePolicy::Allow.effective(Some("allow")),
            LeakagePolicy::Allow
        );
    }

    #[test]
    fn rows_export_example_text() {
        let rows = vec![
            json!({"example_id": "a", "text": "enough characters to be a labeled query span"}),
        ];
        let examples = leakage_examples_from_rows(&rows);
        assert_eq!(examples[0].0, "a");
    }
}
