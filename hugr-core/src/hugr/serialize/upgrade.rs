use thiserror::Error;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum UpgradeError {
    #[error(transparent)]
    Deserialize(#[from] serde_json::Error),

    #[error("Version {0} HUGR serialization format is not supported.")]
    KnownVersionUnsupported(String),

    #[error("Unsupported HUGR serialization format.")]
    UnknownVersionUnsupported,
}

#[cfg(all(test, not(miri)))]
// see serialize::test.
mod test;
