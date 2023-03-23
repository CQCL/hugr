use smol_str::SmolStr;

use super::TypeRow;

/// An opaque type element. Contains an unique identifier and a reference to its definition.
///
/// TODO: We could replace the `Box` with an `Arc` to reduce memory usage,
/// but it adds atomic ops and a serialization-deserialization roundtrip
/// would still generate copies.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CustomType {
    /// Unique identifier of the opaque type.
    id: SmolStr,
    params: TypeRow,
}

impl CustomType {
    pub fn new(id: SmolStr, params: TypeRow) -> Self {
        Self { id, params }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn params(&self) -> &TypeRow {
        &self.params
    }
}

impl PartialEq for CustomType {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for CustomType {}
