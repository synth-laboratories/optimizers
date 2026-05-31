use serde_json::{json, Value};

pub fn usage_from_messages(messages: &[Value], turn_id: &str) -> Option<Value> {
    messages.iter().rev().find_map(|message| {
        let message_turn_id = extract_turn_id(message)?;
        if message_turn_id != turn_id {
            return None;
        }
        usage_from_message(message)
    })
}

pub fn usage_from_message(message: &Value) -> Option<Value> {
    let usage = message
        .pointer("/params/tokenUsage/last")
        .or_else(|| message.pointer("/params/token_usage/last"))
        .or_else(|| message.pointer("/params/turn/usage"))
        .or_else(|| message.pointer("/params/usage"))
        .or_else(|| message.pointer("/result/usage"))?;
    let prompt_tokens = usage_u64(
        usage,
        &[
            "prompt_tokens",
            "input_tokens",
            "inputTokens",
            "promptTokens",
        ],
    );
    let completion_tokens = usage_u64(
        usage,
        &[
            "completion_tokens",
            "output_tokens",
            "outputTokens",
            "completionTokens",
            "reasoning_output_tokens",
            "reasoningOutputTokens",
        ],
    );
    let total_tokens = usage_u64(usage, &["total_tokens", "totalTokens"])
        .max(prompt_tokens.saturating_add(completion_tokens));
    Some(json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }))
}

fn usage_u64(usage: &Value, keys: &[&str]) -> u64 {
    keys.iter()
        .find_map(|key| usage.get(*key).and_then(Value::as_u64))
        .unwrap_or(0)
}

fn extract_turn_id(message: &Value) -> Option<String> {
    message
        .pointer("/result/turn/id")
        .or_else(|| message.pointer("/result/turnId"))
        .or_else(|| message.pointer("/params/turn/id"))
        .or_else(|| message.pointer("/params/turnId"))
        .and_then(Value::as_str)
        .map(str::to_string)
}
