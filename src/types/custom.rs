//! Opaque types, used to represent a user-defined [`Type`].
//!
//! [`Type`]: super::Type
use smol_str::SmolStr;
use std::fmt::{self, Display};

use crate::resource::ResourceId;

use super::{type_param::TypeArg, TypeBound};

/// An opaque type element. Contains the unique identifier of its definition.
#[derive(Debug, PartialEq, Eq, Clone, serde::Serialize, serde::Deserialize)]
pub struct CustomType {
    resource: ResourceId,
    /// Unique identifier of the opaque type.
    /// Same as the corresponding [`TypeDef`]
    ///
    /// [`TypeDef`]: crate::resource::TypeDef
    id: SmolStr,
    /// Arguments that fit the [`TypeParam`]s declared by the typedef
    ///
    /// [`TypeParam`]: super::type_param::TypeParam
    args: Vec<TypeArg>,
    /// The [TypeBound] describing what can be done to instances of this type
    bound: Option<TypeBound>,
}

impl CustomType {
    /// Creates a new opaque type.
    pub fn new(
        id: impl Into<SmolStr>,
        args: impl Into<Vec<TypeArg>>,
        resource: impl Into<ResourceId>,
        bound: Option<TypeBound>,
    ) -> Self {
        Self {
            id: id.into(),
            args: args.into(),
            resource: resource.into(),
            bound,
        }
    }

    /// Creates a new opaque type (constant version, no type arguments)
    pub const fn new_simple(id: SmolStr, resource: ResourceId, bound: Option<TypeBound>) -> Self {
        Self {
            id,
            args: vec![],
            resource,
            bound,
        }
    }

    /// Returns the bound of this [`CustomType`].
    pub fn bound(&self) -> Option<TypeBound> {
        self.bound
    }
}

impl CustomType {
    /// unique name of the type.
    pub fn name(&self) -> &SmolStr {
        &self.id
    }

    /// Type arguments.
    pub fn args(&self) -> &[TypeArg] {
        &self.args
    }

    /// Parent resource.
    pub fn resource(&self) -> &ResourceId {
        &self.resource
    }
}

impl Display for CustomType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}({:?})", self.id, self.args)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use smol_str::SmolStr;

    use crate::types::TypeBound;

    use super::CustomType;

    pub(crate) const EQ_CUST: CustomType = CustomType::new_simple(
        SmolStr::new_inline("MyEqType"),
        SmolStr::new_inline("MyRsrc"),
        Some(TypeBound::Eq),
    );

    pub(crate) const COPYABLE_CUST: CustomType = CustomType::new_simple(
        SmolStr::new_inline("MyCopyableType"),
        SmolStr::new_inline("MyRsrc"),
        Some(TypeBound::Copyable),
    );

    pub(crate) const ANY_CUST: CustomType = CustomType::new_simple(
        SmolStr::new_inline("MyAnyType"),
        SmolStr::new_inline("MyRsrc"),
        None,
    );
}
