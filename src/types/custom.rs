//! Opaque types, used to represent a user-defined [`SimpleType`].
use smol_str::SmolStr;

use super::{ClassicType, SimpleType, TypeRow};

/// An opaque type element. Contains an unique identifier and a reference to its definition.
//
// TODO: We could replace the `Box` with an `Arc` to reduce memory usage,
// but it adds atomic ops and a serialization-deserialization roundtrip
// would still generate copies.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CustomType {
    /// Unique identifier of the opaque type.
    id: SmolStr,
    params: Box<TypeRow>,
}

impl CustomType {
    /// Creates a new opaque type.
    pub fn new(id: SmolStr, params: TypeRow) -> Self {
        Self {
            id,
            params: Box::new(params),
        }
    }

    /// Returns the unique identifier of the opaque type.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the parameters of the opaque type.
    pub fn params(&self) -> &TypeRow {
        &self.params
    }

    /// Returns a [`ClassicType`] containing this opaque type.
    pub const fn classic_type(self) -> ClassicType {
        ClassicType::Opaque(self)
    }
}

impl PartialEq for CustomType {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for CustomType {}

impl From<CustomType> for SimpleType {
    fn from(ty: CustomType) -> Self {
        SimpleType::Classic(ty.classic_type())
    }
}
