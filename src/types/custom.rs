//! Opaque types, used to represent a user-defined [`SimpleType`].
//!
//! [`SimpleType`]: super::SimpleType
use smol_str::SmolStr;
use std::fmt::{self, Display};

use crate::resource::ResourceId;

use super::{type_param::TypeArg, ClassicType, Container, HashableType, SimpleType, TypeTag};

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
    /// The [TypeTag] describing what can be done to instances of this type
    tag: TypeTag,
}

impl CustomType {
    /// Creates a new opaque type.
    pub fn new(
        id: impl Into<SmolStr>,
        args: impl Into<Vec<TypeArg>>,
        resource: impl Into<ResourceId>,
        tag: TypeTag,
    ) -> Self {
        Self {
            id: id.into(),
            args: args.into(),
            resource: resource.into(),
            tag,
        }
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

/// This parallels [SimpleType::new_tuple] and [SimpleType::new_sum]
impl From<CustomType> for SimpleType {
    fn from(value: CustomType) -> Self {
        match value.tag {
            TypeTag::Simple => SimpleType::Qontainer(Container::Opaque(value)),
            TypeTag::Classic => ClassicType::Container(Container::Opaque(value)).into(),
            TypeTag::Hashable => HashableType::Container(Container::Opaque(value)).into(),
        }
    }
}
