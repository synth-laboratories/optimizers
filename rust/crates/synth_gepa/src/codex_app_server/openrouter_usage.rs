use serde_json::{json, Map, Value};

pub(super) fn normalize(
    model: &str,
    mut usage: Map<String, Value>,
    reported_cost: Option<f64>,
) -> Value {
    usage.insert("provider".into(), Value::String("openrouter".into()));
    usage.insert("model".into(), Value::String(model.into()));
    if reported_cost.is_none() {
        let cost = usage
            .get("cost")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value >= 0.0);
        if let Some(cost) = cost {
            usage.insert("cost_usd".into(), json!(cost));
            usage.insert(
                "cost_source".into(),
                Value::String("openrouter_provider_billed".into()),
            );
        }
    }
    Value::Object(usage)
}
