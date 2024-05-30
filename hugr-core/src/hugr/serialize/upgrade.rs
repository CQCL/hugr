use thiserror::Error;

#[derive(Debug, Error)]
pub enum UpgradeError {
    #[error(transparent)]
    Deserialize(#[from] serde_json::Error),

    #[error("Version {0} HUGR serialization format is not supported.")]
    KnownVersionUnsupported(String),

    #[error("Unsupported HUGR serialization format.")]
    UnknownVersionUnsupported,
}

pub fn v1_to_v2(_: serde_json::Value) -> Result<serde_json::Value, UpgradeError> {
    todo!()
}
