use serde::de::Error;
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

pub fn v1_to_v2(mut input: serde_json::Value) -> Result<serde_json::Value, UpgradeError> {
    let input_obj = input
        .as_object_mut()
        .ok_or(serde_json::Error::custom("Value is not an object"))?;
    let nodes = input_obj
        .get_mut("nodes")
        .ok_or(serde_json::Error::custom("No nodes field"))?
        .as_array_mut()
        .ok_or(serde_json::Error::custom("nodes is not an array"))?;

    let mut hierarchy = vec![];
    for node in nodes.iter_mut() {
        let node_obj = node
            .as_object_mut()
            .ok_or(serde_json::Error::custom("Node is not an object"))?;
        let parent = node_obj
            .remove("parent")
            .ok_or(serde_json::Error::custom("No parent field"))?;
        hierarchy.push(parent);
    }
    input_obj.insert("hierarchy".to_string(), hierarchy.into());
    Ok(input)
}

#[cfg(test)]
mod test;
