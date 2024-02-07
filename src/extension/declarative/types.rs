//! Declarative type definitions.
//!
//! This module defines a YAML schema for defining types in a declarative way.
//!
//! See the [specification] and [`ExtensionSetDeclaration`] for more details.
//!
//! [specification]: https://github.com/CQCL/hugr/blob/main/specification/hugr.md#declarative-format
//! [`ExtensionSetDeclaration`]: super::ExtensionSetDeclaration

use crate::types::{TypeBound, TypeName};

use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize};

/// A declarative type definition.
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub(super) struct TypeDeclaration {
    /// The name of the type.
    name: TypeName,
    /// The [`TypeBound`] describing what can be done to instances of this type.
    /// Options are `Eq`, `Copyable`, or `Any`.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    bound: TypeBoundDeclaration,
    /// A list of type parameters for this type.
    ///
    /// Each element in the list is a 2-element list, where the first element is
    /// the human-readable name of the type parameter, and the second element is
    /// the type id.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    params: Vec<TypeParamDeclaration>,
}

/// A declarative TypeBound definition.
///
/// Equivalent to a [`TypeBound`]. Provides human-friendly serialization, using
/// the full names.
#[derive(
    Debug, Copy, Clone, Serialize, Deserialize, Hash, PartialEq, Eq, Default, derive_more::Display,
)]
enum TypeBoundDeclaration {
    /// The equality operation is valid on this type.
    Eq,
    /// The type can be copied in the program.
    Copyable,
    /// No bound on the type.
    #[default]
    Any,
}

impl From<TypeBoundDeclaration> for TypeBound {
    fn from(bound: TypeBoundDeclaration) -> Self {
        match bound {
            TypeBoundDeclaration::Eq => Self::Eq,
            TypeBoundDeclaration::Copyable => Self::Copyable,
            TypeBoundDeclaration::Any => Self::Any,
        }
    }
}

impl From<TypeBound> for TypeBoundDeclaration {
    fn from(bound: TypeBound) -> Self {
        match bound {
            TypeBound::Eq => Self::Eq,
            TypeBound::Copyable => Self::Copyable,
            TypeBound::Any => Self::Any,
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
    description: Option<String>,
    /// The parameter type.
    type_name: TypeName,
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
