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

#[allow(unused_mut)]
pub fn v1_to_v2(mut input: serde_json::Value) -> Result<serde_json::Value, UpgradeError> {
    Ok(input)
}

#[cfg(test)]
mod test;
