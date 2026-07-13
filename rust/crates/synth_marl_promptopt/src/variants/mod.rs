mod coma;
mod ic3net;
mod imac;
mod rode;

use synth_optimizer_platform::{OptimizerError, Result};

use crate::strategy::MarlStrategy;

pub use coma::ComaStrategy;
pub use ic3net::Ic3NetStrategy;
pub use imac::ImacStrategy;
pub use rode::RodeStrategy;

pub fn strategy_by_name(name: &str) -> Result<Box<dyn MarlStrategy>> {
    match name.trim().to_ascii_lowercase().as_str() {
        "coma" | "counterfactual_credit" => Ok(Box::new(ComaStrategy)),
        "ic3net" | "speak_gate" => Ok(Box::new(Ic3NetStrategy)),
        "imac" | "information_bottleneck" => Ok(Box::new(ImacStrategy)),
        "rode" | "role_hierarchy" => Ok(Box::new(RodeStrategy)),
        other => Err(OptimizerError::Config(format!(
            "unknown MARL prompt-optimizer variant {other:?}; expected coma, ic3net, imac, or rode"
        ))),
    }
}
