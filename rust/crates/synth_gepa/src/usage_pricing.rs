//! Static USD pricing for usage that APIs return without a billed `cost_usd`.
//!
//! A `cost_usd` of 0.0 with nonzero tokens is unpriced, not free. Callers should
//! either apply this catalog or leave `cost_source=unpriced` rather than treat
//! the zero as a USD measurement.

use serde_json::{json, Map, Value};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelRates {
    pub input_usd_per_million: f64,
    pub cached_input_usd_per_million: f64,
    pub output_usd_per_million: f64,
}

impl ModelRates {
    pub const fn new(input: f64, cached: f64, output: f64) -> Self {
        Self {
            input_usd_per_million: input,
            cached_input_usd_per_million: cached,
            output_usd_per_million: output,
        }
    }

    pub fn price_usd(self, prompt_tokens: u64, completion_tokens: u64, cached_prompt_tokens: u64) -> f64 {
        let cached = cached_prompt_tokens.min(prompt_tokens);
        let billable_prompt = prompt_tokens.saturating_sub(cached);
        billable_prompt as f64 * self.input_usd_per_million / 1_000_000.0
            + cached as f64 * self.cached_input_usd_per_million / 1_000_000.0
            + completion_tokens as f64 * self.output_usd_per_million / 1_000_000.0
    }
}

pub fn normalize_model_id(model: &str) -> String {
    model
        .trim()
        .trim_start_matches("openai/")
        .trim_start_matches("openrouter/")
        .to_ascii_lowercase()
}

pub fn rates_for(model: &str, provider: Option<&str>) -> Option<ModelRates> {
    let model = normalize_model_id(model);
    let provider = provider.unwrap_or("").trim().to_ascii_lowercase();
    if provider == "deepseek" || model.contains("deepseek") {
        return Some(ModelRates::new(0.27, 0.27, 1.10));
    }
    Some(match model.as_str() {
        "gpt-4.1-nano" => ModelRates::new(0.10, 0.025, 0.40),
        "gpt-4.1-mini" => ModelRates::new(0.40, 0.10, 1.60),
        "gpt-4.1" => ModelRates::new(2.00, 0.50, 8.00),
        "gpt-5.4-nano" => ModelRates::new(0.20, 0.02, 1.25),
        "gpt-5.4-mini" => ModelRates::new(0.75, 0.075, 4.50),
        "gpt-5.6-luna" => ModelRates::new(0.20, 0.02, 1.20),
        "gpt-5.6-terra" => ModelRates::new(2.00, 0.20, 12.00),
        "gpt-5.6-sol" => ModelRates::new(5.00, 0.50, 30.00),
        "x-ai/grok-4.3" | "grok-4.3" => ModelRates::new(1.25, 0.20, 2.50),
        "nvidia/nemotron-3.5-lightning" | "nemotron-3.5-lightning" => {
            ModelRates::new(0.08, 0.04, 0.20)
        }
        _ => return None,
    })
}

pub fn price_usd(
    model: &str,
    provider: Option<&str>,
    prompt_tokens: u64,
    completion_tokens: u64,
    cached_prompt_tokens: u64,
) -> Option<f64> {
    rates_for(model, provider).map(|rates| {
        rates.price_usd(prompt_tokens, completion_tokens, cached_prompt_tokens)
    })
}

fn usage_u64(usage: &Map<String, Value>, keys: &[&str]) -> u64 {
    keys.iter()
        .find_map(|key| {
            usage.get(*key).and_then(|value| {
                value
                    .as_u64()
                    .or_else(|| value.as_f64().map(|number| number.max(0.0) as u64))
                    .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
            })
        })
        .unwrap_or(0)
}

fn usage_f64(usage: &Map<String, Value>, key: &str) -> Option<f64> {
    usage.get(key).and_then(|value| {
        value
            .as_f64()
            .or_else(|| value.as_u64().map(|number| number as f64))
            .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
    })
}

fn nested_cached_tokens(usage: &Map<String, Value>, key: &str) -> u64 {
    usage
        .get(key)
        .and_then(Value::as_object)
        .and_then(|details| details.get("cached_tokens"))
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_f64().map(|number| number.max(0.0) as u64))
        })
        .unwrap_or(0)
}

fn cached_prompt_tokens(usage: &Map<String, Value>) -> u64 {
    let direct = usage_u64(usage, &["cached_prompt_tokens", "cached_tokens"]);
    if direct > 0 {
        return direct;
    }
    let nested = nested_cached_tokens(usage, "prompt_tokens_details")
        .max(nested_cached_tokens(usage, "input_tokens_details"));
    nested
}

pub fn billed_cost_usd(usage: &Map<String, Value>) -> Option<f64> {
    let cost = usage_f64(usage, "cost_usd")
        .or_else(|| usage_f64(usage, "spend_usd"))
        .or_else(|| usage_f64(usage, "cost"))?;
    if !cost.is_finite() {
        return None;
    }
    if cost > 0.0 {
        return Some(cost);
    }
    let prompt = usage_u64(usage, &["prompt_tokens", "input_tokens"]);
    let completion = usage_u64(usage, &["completion_tokens", "output_tokens"]);
    if cost == 0.0 && prompt == 0 && completion == 0 {
        return Some(0.0);
    }
    None
}

/// Fill `cost_usd` from a billed value or the static catalog. Zero tokens stay
/// zero. Unknown models with tokens stay `0.0` and are marked `unpriced`.
pub fn ensure_priced_usage(
    usage: &mut Map<String, Value>,
    model: &str,
    provider: Option<&str>,
) -> f64 {
    if !model.trim().is_empty() && usage.get("model").and_then(Value::as_str).is_none() {
        usage.insert("model".to_string(), Value::String(model.to_string()));
    }
    if let Some(provider) = provider.filter(|value| !value.trim().is_empty()) {
        if usage.get("provider").and_then(Value::as_str).is_none() {
            usage.insert("provider".to_string(), Value::String(provider.to_string()));
        }
    }
    if let Some(cost) = billed_cost_usd(usage) {
        if cost > 0.0 && usage.get("cost_source").is_none() {
            usage.insert(
                "cost_source".to_string(),
                Value::String("provider_billed".to_string()),
            );
        }
        return cost;
    }
    let prompt = usage_u64(usage, &["prompt_tokens", "input_tokens"]);
    let completion = usage_u64(usage, &["completion_tokens", "output_tokens"]);
    let cached = cached_prompt_tokens(usage).min(prompt);
    if prompt == 0 && completion == 0 {
        usage.insert("cost_usd".to_string(), json!(0.0));
        usage.insert(
            "cost_source".to_string(),
            Value::String("no_tokens".to_string()),
        );
        return 0.0;
    }
    match rates_for(model, provider) {
        Some(rates) => {
            let cost = rates.price_usd(prompt, completion, cached);
            usage.insert("cost_usd".to_string(), json!(cost));
            usage.insert(
                "cost_source".to_string(),
                Value::String(format!("static_price:{}", normalize_model_id(model))),
            );
            usage.insert(
                "cost_pricing".to_string(),
                json!({
                    "input_usd_per_million": rates.input_usd_per_million,
                    "cached_input_usd_per_million": rates.cached_input_usd_per_million,
                    "output_usd_per_million": rates.output_usd_per_million,
                }),
            );
            cost
        }
        None => {
            usage.insert(
                "cost_source".to_string(),
                Value::String("unpriced".to_string()),
            );
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_cost_with_tokens_is_unpriced_until_catalog_applies() {
        let mut usage = Map::new();
        usage.insert("prompt_tokens".to_string(), json!(1_000_000));
        usage.insert("completion_tokens".to_string(), json!(1_000_000));
        usage.insert("cost_usd".to_string(), json!(0.0));
        let cost = ensure_priced_usage(&mut usage, "gpt-4.1-nano", Some("openai"));
        assert!((cost - 0.50).abs() < 1e-9, "cost={cost}");
        assert_eq!(
            usage.get("cost_source").and_then(Value::as_str),
            Some("static_price:gpt-4.1-nano")
        );
    }

    #[test]
    fn billed_cost_wins_over_catalog() {
        let mut usage = Map::new();
        usage.insert("prompt_tokens".to_string(), json!(1_000_000));
        usage.insert("completion_tokens".to_string(), json!(0));
        usage.insert("cost_usd".to_string(), json!(12.5));
        let cost = ensure_priced_usage(&mut usage, "gpt-4.1-nano", Some("openai"));
        assert!((cost - 12.5).abs() < 1e-9);
        assert_eq!(
            usage.get("cost_source").and_then(Value::as_str),
            Some("provider_billed")
        );
    }

    #[test]
    fn luna_and_nemotron_have_catalog_rates() {
        assert_eq!(
            price_usd("gpt-5.6-luna", Some("openai"), 1_000_000, 1_000_000, 0),
            Some(1.40)
        );
        assert_eq!(
            price_usd(
                "nvidia/nemotron-3.5-lightning",
                Some("openrouter"),
                1_000_000,
                1_000_000,
                0
            ),
            Some(0.28)
        );
    }
}
