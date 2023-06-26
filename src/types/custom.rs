//! Opaque types, used to represent a user-defined [`SimpleType`].
//!
//! [`SimpleType`]: super::SimpleType
use smol_str::SmolStr;
use std::fmt::{self, Display};

use super::{type_param::TypeArg, ClassicType};

/// An opaque type element. Contains an unique identifier and a reference to its definition.
//
// TODO: We could replace the `Box` with an `Arc` to reduce memory usage,
// but it adds atomic ops and a serialization-deserialization roundtrip
// would still generate copies.
#[derive(Debug, PartialEq, Eq, Clone, serde::Serialize, serde::Deserialize)]
pub struct CustomType {
    /// Unique identifier of the opaque type.
    /// Same as the corresponding [`TypeDef`]
    ///
    /// [`TypeDef`]: crate::resource::TypeDef
    pub id: SmolStr,
    /// Arguments that fit the [`TypeParam`]s declared by the typedef
    ///
    /// [`TypeParam`]: super::type_param::TypeParam
    pub params: Vec<TypeArg>,
}

impl CustomType {
    /// Creates a new opaque type.
    pub fn new(id: SmolStr, params: impl Into<Vec<TypeArg>>) -> Self {
        Self {
            id,
            params: params.into(),
        }
    }

    /// Returns the unique identifier of the opaque type.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns a [`ClassicType`] containing this opaque type.
    pub const fn classic_type(self) -> ClassicType {
        ClassicType::Opaque(self)
    }
}

impl Display for CustomType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}({:?})", self.id, self.params)
    }
}

impl From<CustomType> for ClassicType {
    fn from(ty: CustomType) -> Self {
        ty.classic_type()
    }
}
