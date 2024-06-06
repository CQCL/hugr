//! Declarative type definitions.
//!
//! This module defines a YAML schema for defining types in a declarative way.
//!
//! See the [specification] and [`ExtensionSetDeclaration`] for more details.
//!
//! [specification]: https://github.com/CQCL/hugr/blob/main/specification/hugr.md#declarative-format
//! [`ExtensionSetDeclaration`]: super::ExtensionSetDeclaration

use crate::extension::{TypeDef, TypeDefBound};
use crate::types::type_param::TypeParam;
use crate::types::{CustomType, TypeBound, TypeName};
use crate::Extension;

use serde::{Deserialize, Serialize};

use super::{DeclarationContext, ExtensionDeclarationError};

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
    ///
    /// Types in the definition will be resolved using the extensions in `scope`
    /// and the current extension.
    pub fn register<'ext>(
        &self,
        ext: &'ext mut Extension,
        ctx: DeclarationContext<'_>,
    ) -> Result<&'ext TypeDef, ExtensionDeclarationError> {
        let params = self
            .params
            .iter()
            .map(|param| param.make_type_param(ext, ctx))
            .collect::<Result<Vec<TypeParam>, _>>()?;
        let type_def = ext.add_type(
            self.name.clone(),
            params,
            self.description.clone(),
            self.bound.into(),
        )?;
        Ok(type_def)
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
/// Either a type, or a 2-element list containing a human-readable name and a type id.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
enum TypeParamDeclaration {
    /// Just the type id.
    Type(TypeName),
    /// A 2-tuple containing a human-readable name and a type id.
    WithDescription(String, TypeName),
}

impl TypeParamDeclaration {
    /// Create a [`TypeParam`] from this declaration.
    ///
    /// Resolves any type ids using both the current extension and any other in `scope`.
    ///
    /// TODO: Only non-parametric opaque types are supported for now.
    /// TODO: The parameter description is currently ignored.
    pub fn make_type_param(
        &self,
        extension: &Extension,
        ctx: DeclarationContext<'_>,
    ) -> Result<TypeParam, ExtensionDeclarationError> {
        let instantiate_type = |ty: &TypeDef| -> Result<CustomType, ExtensionDeclarationError> {
            match ty.params() {
                [] => Ok(ty.instantiate([]).unwrap()),
                _ => Err(ExtensionDeclarationError::ParametricTypeParameter {
                    ext: extension.name().clone(),
                    ty: self.type_name().clone(),
                }),
            }
        };

        // First try the previously defined types in the current extension.
        if let Some(ty) = extension.get_type(self.type_name()) {
            return Ok(TypeParam::Opaque {
                ty: instantiate_type(ty)?,
            });
        }

        // Try every extension in scope.
        //
        // TODO: Can we resolve the extension id from the type name instead?
        for ext in ctx.scope.iter() {
            if let Some(ty) = ctx
                .registry
                .get(ext)
                .and_then(|ext| ext.get_type(self.type_name()))
            {
                return Ok(TypeParam::Opaque {
                    ty: instantiate_type(ty)?,
                });
            }
        }

        Err(ExtensionDeclarationError::MissingType {
            ext: extension.name().clone(),
            ty: self.type_name().clone(),
        })
    }

    /// Returns the type name of this type parameter.
    fn type_name(&self) -> &TypeName {
        match self {
            Self::Type(ty) => ty,
            Self::WithDescription(_, ty) => ty,
        }
    }
}
