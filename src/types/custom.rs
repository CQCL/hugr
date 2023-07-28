//! Opaque types, used to represent a user-defined [`SimpleType`].
//!
//! [`SimpleType`]: super::SimpleType
use smol_str::SmolStr;
use std::fmt::{self, Display};

use super::{type_param::TypeArg, ClassicType, Container};

/// An opaque type element. Contains the unique identifier of its definition.
#[derive(Debug, PartialEq, Eq, Clone, serde::Serialize, serde::Deserialize)]
pub struct CustomType {
    /// Unique identifier of the opaque type.
    /// Same as the corresponding [`TypeDef`]
    ///
    /// [`TypeDef`]: crate::resource::TypeDef
    id: SmolStr,
    /// Arguments that fit the [`TypeParam`]s declared by the typedef
    ///
    /// [`TypeParam`]: super::type_param::TypeParam
    args: Vec<TypeArg>,
}

impl CustomType {
    /// Creates a new opaque type.
    pub fn new(id: impl Into<SmolStr>, args: impl Into<Vec<TypeArg>>) -> Self {
        Self {
            id: id.into(),
            args: args.into(),
        }
    }

    /// Creates a new opaque type with no parameters
    pub const fn new_simple(id: SmolStr) -> Self {
        Self { id, args: vec![] }
    }

    /// Returns the unique identifier of the opaque type.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the arguments of the opaque type.
    pub fn args(&self) -> &[TypeArg] {
        &self.args
    }

    /// Returns a [`ClassicType`] containing this opaque type.
    pub const fn classic_type(self) -> ClassicType {
        ClassicType::Container(Container::Opaque(self))
    }
}

impl Display for CustomType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}({:?})", self.id, self.args)
    }
}

impl From<CustomType> for ClassicType {
    fn from(ty: CustomType) -> Self {
        ty.classic_type()
    }
}
