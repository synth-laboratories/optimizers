pub mod campaign;
pub mod candidate;
pub mod config;
pub mod executor;
pub mod proposer;
pub mod review;
pub mod runtime;
pub mod scoring;

pub use campaign::{CampaignBinding, DebriefCampaignManifest};
pub use candidate::{
    MapoCandidate, MapoProtocolConfig, MapoRolloutRecord, MapoSharedContextConfig,
};
pub use config::{
    MapoAlgorithmConfig, MapoConfig, MapoEvidenceConfig, MapoExecutionOptions, MapoRunConfig,
};
pub use review::{build_mapo_review_rows, MapoReviewRow};
pub use runtime::{execute_mapo_with_options, MapoRunResult, MAPO_ALGORITHM_ID};
pub use scoring::MapoScore;
