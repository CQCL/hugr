//! Declarative type definitions.
//!
//! This module defines a YAML schema for defining types in a declarative way.
//!
//! See the [specification] and [`ExtensionSetDeclaration`] for more details.
//!
//! [specification]: https://github.com/CQCL/hugr/blob/main/specification/hugr.md#declarative-format
//! [`ExtensionSetDeclaration`]: super::ExtensionSetDeclaration

use crate::extension::{ExtensionBuildError, TypeDef, TypeDefBound};
use crate::types::type_param::TypeParam;
use crate::types::{TypeBound, TypeName};

use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize};

/// A declarative type definition.
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub(super) struct TypeDeclaration {
    /// The name of the type.
    name: TypeName,
    /// A description for the type.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    description: String,
    /// The type bound describing what can be done to instances of this type.
    /// Options are `Eq`, `Copyable`, or `Any`.
    ///
    /// See [`TypeBound`] and [`TypeDefBound`].
    ///
    /// TODO: Derived bounds from the parameters (see [`TypeDefBound`]) are not yet supported.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    bound: TypeDefBoundDeclaration,
    /// A list of type parameters for this type.
    ///
    /// Each element in the list is a 2-element list, where the first element is
    /// the human-readable name of the type parameter, and the second element is
    /// the type id.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    params: Vec<TypeParamDeclaration>,
}

impl TypeDeclaration {
    /// Register this type in the given extension.
    pub fn register<'ext>(
        &self,
        ext: &'ext mut crate::Extension,
    ) -> Result<&'ext TypeDef, ExtensionBuildError> {
        let params = self
            .params
            .iter()
            .map(TypeParamDeclaration::make_type_param)
            .collect();
        ext.add_type(
            self.name.clone(),
            params,
            self.description.clone(),
            self.bound.into(),
        )
    }
}

/// A declarative TypeBound definition.
///
/// Equivalent to a [`TypeDefBound`]. Provides human-friendly serialization, using
/// the full names.
///
/// TODO: Support derived bounds
#[derive(
    Debug, Copy, Clone, Serialize, Deserialize, Hash, PartialEq, Eq, Default, derive_more::Display,
)]
enum TypeDefBoundDeclaration {
    /// The equality operation is valid on this type.
    Eq,
    /// The type can be copied in the program.
    Copyable,
    /// No bound on the type.
    #[default]
    Any,
}

impl From<TypeDefBoundDeclaration> for TypeDefBound {
    fn from(bound: TypeDefBoundDeclaration) -> Self {
        match bound {
            TypeDefBoundDeclaration::Eq => Self::Explicit(TypeBound::Eq),
            TypeDefBoundDeclaration::Copyable => Self::Explicit(TypeBound::Copyable),
            TypeDefBoundDeclaration::Any => Self::Explicit(TypeBound::Any),
        }
    }
}

/// A declarative type parameter definition.
///
/// Serialized as a 2-element list, where the first element is an optional
/// human-readable name of the type parameter, and the second element is the
/// type id.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct TypeParamDeclaration {
    /// The name of the parameter. May be `null`.
    ///
    /// TODO: This field is ignored, there's no place to put it in a `TypeParam`.
    description: Option<String>,
    /// The parameter type.
    type_name: TypeName,
}

impl TypeParamDeclaration {
    /// Create a [`TypeParam`] from this declaration.
    ///
    /// Only opaque types are supported for now.
    pub fn make_type_param(&self) -> TypeParam {
        // TODO: We have to resolve the type id to a real type.
        let _ty = unimplemented!();
        //TypeParam::Opaque { ty }
    }
}

impl Serialize for TypeParamDeclaration {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(2))?;
        seq.serialize_element(&self.description)?;
        seq.serialize_element(&self.type_name)?;
        seq.end()
    }
}

impl<'de> Deserialize<'de> for TypeParamDeclaration {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct TypeParamVisitor;
        const EXPECTED_MSG: &str = "a 2-element list containing a type parameter name and id";

        impl<'de> serde::de::Visitor<'de> for TypeParamVisitor {
            type Value = TypeParamDeclaration;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str(EXPECTED_MSG)
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let description = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &EXPECTED_MSG))?;
                let type_name = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &EXPECTED_MSG))?;
                Ok(TypeParamDeclaration {
                    description,
                    type_name,
                })
            }
        }

        deserializer.deserialize_seq(TypeParamVisitor)
    }
}
