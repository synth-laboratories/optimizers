//! Provider wire-cost normalization (SYN-5275).
//!
//! Providers report billed spend under different keys and encodings:
//! normalized receipts use `usage.cost_usd`, OpenRouter uses `usage.cost`, and
//! some gateways send the amount as a decimal string. Every reader that turns a
//! wire payload into settled spend goes through [`read_wire_cost`] so that a
//! finite, non-negative amount (including a real zero) is known, and an absent
//! or invalid amount stays unknown. Unknown spend is never coerced to zero here;
//! the hard budget decides what to do with it.

use serde_json::Value;

/// Normalized field read by the runtime and the budget ledger.
pub(crate) const COST_USD_FIELD: &str = "cost_usd";
/// OpenRouter's provider-billed field.
pub(crate) const PROVIDER_COST_FIELD: &str = "cost";
/// Diagnostic field written when the provider sent a cost the parser refused.
pub(crate) const COST_UNKNOWN_REASON_FIELD: &str = "cost_unknown_reason";
/// Reason code for a provider cost that is present but not a finite non-negative amount.
pub(crate) const PROVIDER_COST_INVALID: &str = "provider_cost_invalid";

/// Typed reading of one wire cost field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum WireCost {
    /// The field is missing or JSON `null`.
    Absent,
    /// A finite, non-negative USD amount. Zero is a real, known cost.
    Known(f64),
    /// The field is present but negative, non-finite, or not a number.
    Invalid,
}

impl WireCost {
    pub(crate) fn known(self) -> Option<f64> {
        match self {
            Self::Known(value) => Some(value),
            Self::Absent | Self::Invalid => None,
        }
    }
}

/// Classify one wire value.
pub(crate) fn read_wire_cost(value: Option<&Value>) -> WireCost {
    match value {
        None | Some(Value::Null) => WireCost::Absent,
        Some(Value::Number(number)) => number.as_f64().map_or(WireCost::Invalid, classify),
        Some(Value::String(text)) => text
            .trim()
            .parse::<f64>()
            .map_or(WireCost::Invalid, classify),
        Some(_) => WireCost::Invalid,
    }
}

/// Provider-reported spend from a usage payload: the normalized `cost_usd`
/// wins when it is known, otherwise OpenRouter's `cost`.
pub(crate) fn provider_cost_usd(usage: &Value) -> Option<f64> {
    read_wire_cost(usage.get(COST_USD_FIELD))
        .known()
        .or_else(|| read_wire_cost(usage.get(PROVIDER_COST_FIELD)).known())
}

fn classify(value: f64) -> WireCost {
    if !value.is_finite() || value < 0.0 {
        return WireCost::Invalid;
    }
    // Fold -0.0 into 0.0 so receipts never carry a negative-zero amount.
    WireCost::Known(if value == 0.0 { 0.0 } else { value })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn finite_non_negative_numbers_and_strings_are_known() {
        assert_eq!(
            read_wire_cost(Some(&json!(0.0157))),
            WireCost::Known(0.0157)
        );
        assert_eq!(
            read_wire_cost(Some(&json!(" 0.004 "))),
            WireCost::Known(0.004)
        );
        assert_eq!(read_wire_cost(Some(&json!(3))), WireCost::Known(3.0));
    }

    #[test]
    fn real_zero_is_known_and_negative_zero_folds() {
        assert_eq!(read_wire_cost(Some(&json!(0))), WireCost::Known(0.0));
        assert_eq!(read_wire_cost(Some(&json!("0"))), WireCost::Known(0.0));
        let WireCost::Known(folded) = read_wire_cost(Some(&json!(-0.0))) else {
            panic!("negative zero must be a known zero");
        };
        assert!(folded.is_sign_positive());
    }

    #[test]
    fn absent_and_null_are_absent() {
        assert_eq!(read_wire_cost(None), WireCost::Absent);
        assert_eq!(read_wire_cost(Some(&Value::Null)), WireCost::Absent);
    }

    #[test]
    fn negative_non_finite_and_non_numeric_are_invalid() {
        for value in [
            json!(-1.0),
            json!("-0.5"),
            json!("NaN"),
            json!("inf"),
            json!("n/a"),
            json!(""),
            json!(true),
            json!({"usd": 1.0}),
            json!([0.1]),
        ] {
            assert_eq!(read_wire_cost(Some(&value)), WireCost::Invalid, "{value}");
        }
    }

    #[test]
    fn provider_cost_prefers_known_cost_usd_then_cost() {
        assert_eq!(
            provider_cost_usd(&json!({"cost_usd": 0.02, "cost": 0.01})),
            Some(0.02)
        );
        assert_eq!(
            provider_cost_usd(&json!({"cost_usd": 0.0, "cost": 0.01})),
            Some(0.0)
        );
        assert_eq!(
            provider_cost_usd(&json!({"cost_usd": -1, "cost": "0.01"})),
            Some(0.01)
        );
        assert_eq!(
            provider_cost_usd(&json!({"cost_usd": null, "cost": 0})),
            Some(0.0)
        );
        assert_eq!(provider_cost_usd(&json!({"cost": "n/a"})), None);
        assert_eq!(provider_cost_usd(&json!({"total_tokens": 10})), None);
    }
}
