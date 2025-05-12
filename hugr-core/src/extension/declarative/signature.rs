//! Declarative signature definitions.
//!
//! This module defines a YAML schema for defining the signature of an operation in a declarative way.
//!
//! See the [specification] and [`ExtensionSetDeclaration`] for more details.
//!
//! [specification]: https://github.com/CQCL/hugr/blob/main/specification/hugr.md#declarative-format
//! [`ExtensionSetDeclaration`]: super::ExtensionSetDeclaration

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::Extension;
use crate::extension::prelude::PRELUDE_ID;
use crate::extension::{SignatureFunc, TypeDef};
use crate::types::type_param::TypeParam;
use crate::types::{CustomType, FuncValueType, PolyFuncTypeRV, Type, TypeRowRV};

use super::{DeclarationContext, ExtensionDeclarationError};

/// A declarative operation signature definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct SignatureDeclaration {
    /// The inputs to the operation.
    inputs: Vec<SignaturePortDeclaration>,
    /// The outputs of the operation.
    outputs: Vec<SignaturePortDeclaration>,
}

impl SignatureDeclaration {
    /// Register this signature in the given extension.
    pub fn make_signature(
        &self,
        ext: &Extension,
        ctx: DeclarationContext<'_>,
        op_params: &[TypeParam],
    ) -> Result<SignatureFunc, ExtensionDeclarationError> {
        let make_type_row =
            |v: &[SignaturePortDeclaration]| -> Result<TypeRowRV, ExtensionDeclarationError> {
                let types = v
                    .iter()
                    .map(|port_decl| port_decl.make_types(ext, ctx, op_params))
                    .flatten_ok()
                    .collect::<Result<Vec<Type>, _>>()?;
                Ok(types.into())
            };

        let body = FuncValueType {
            input: make_type_row(&self.inputs)?,
            output: make_type_row(&self.outputs)?,
        };

        let poly_func = PolyFuncTypeRV::new(op_params, body);
        Ok(poly_func.into())
    }
}

/// A declarative definition for a number of ports in a signature's input or output.
///
/// Serialized as a single type, or as a 2 or 3-element lists.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
enum SignaturePortDeclaration {
    /// A single port type.
    Type(TypeDeclaration),
    /// A 2-tuple with the type and a repetition declaration.
    TypeRepeat(TypeDeclaration, PortRepetitionDeclaration),
    /// A 3-tuple with a description, a type declaration, and a repetition declaration.
    DescriptionTypeRepeat(String, TypeDeclaration, PortRepetitionDeclaration),
}

impl SignaturePortDeclaration {
    /// Return an iterator with the types for this port declaration.
    fn make_types(
        &self,
        ext: &Extension,
        ctx: DeclarationContext<'_>,
        op_params: &[TypeParam],
    ) -> Result<impl Iterator<Item = Type>, ExtensionDeclarationError> {
        let n: usize = match self.repeat() {
            PortRepetitionDeclaration::Count(n) => *n,
            PortRepetitionDeclaration::Parameter(parametric_repetition) => {
                return Err(ExtensionDeclarationError::UnsupportedPortRepetition {
                    ext: ext.name().clone(),
                    parametric_repetition: parametric_repetition.clone(),
                });
            }
        };

        let ty = self.type_decl().make_type(ext, ctx, op_params)?;
        let ty = Type::new_extension(ty);

        Ok(itertools::repeat_n(ty, n))
    }

    /// Get the type declaration for this port.
    fn type_decl(&self) -> &TypeDeclaration {
        match self {
            SignaturePortDeclaration::Type(ty) => ty,
            SignaturePortDeclaration::TypeRepeat(ty, _) => ty,
            SignaturePortDeclaration::DescriptionTypeRepeat(_, ty, _) => ty,
        }
    }

    /// Get the repetition declaration for this port.
    fn repeat(&self) -> &PortRepetitionDeclaration {
        static DEFAULT_REPEAT: PortRepetitionDeclaration = PortRepetitionDeclaration::Count(1);
        match self {
            SignaturePortDeclaration::DescriptionTypeRepeat(_, _, repeat) => repeat,
            SignaturePortDeclaration::TypeRepeat(_, repeat) => repeat,
            _ => &DEFAULT_REPEAT,
        }
    }
}

/// A number of repetitions for a signature's port definition.
///
/// This value must be a number, indicating a repetition of the port that amount of times.
///
/// Generic expressions are not yet supported.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
enum PortRepetitionDeclaration {
    /// A constant number of repetitions for the port definition.
    Count(usize),
    /// An (integer) operation parameter identifier to use as the number of repetitions.
    Parameter(SmolStr),
}

impl Default for PortRepetitionDeclaration {
    fn default() -> Self {
        PortRepetitionDeclaration::Count(1)
    }
}

/// A type declaration used in signatures.
///
/// TODO: The spec definition is more complex than just a type identifier,
/// we should be able to support expressions like:
///
/// - `Q`
/// - `Array<i>(Array<j>(F64))`
/// - `Function[r](USize -> USize)`
/// - `Opaque(complex_matrix,i,j)`
///
/// Note that `Q` is not the name used for a qubit in the prelude.
///
/// For now, we just hard-code some basic types.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
struct TypeDeclaration(
    /// The encoded type description.
    String,
);

impl TypeDeclaration {
    /// Parse the type represented by this declaration.
    ///
    /// Currently hard-codes some basic types.
    ///
    /// TODO: Support arbitrary types.
    /// TODO: Support parametric types.
    pub fn make_type(
        &self,
        ext: &Extension,
        ctx: DeclarationContext<'_>,
        _op_params: &[TypeParam],
    ) -> Result<CustomType, ExtensionDeclarationError> {
        let Some(type_def) = self.resolve_type(ext, ctx) else {
            return Err(ExtensionDeclarationError::UnknownType {
                ext: ext.name().clone(),
                ty: self.0.clone(),
            });
        };

        // The hard-coded types are not parametric.
        assert!(type_def.params().is_empty());
        let op = type_def.instantiate(&[]).unwrap();

        Ok(op)
    }

    /// Resolve a type name to a type definition.
    fn resolve_type<'a>(
        &'a self,
        ext: &'a Extension,
        ctx: DeclarationContext<'a>,
    ) -> Option<&'a TypeDef> {
        // The prelude is always in scope.
        debug_assert!(ctx.scope.contains(&PRELUDE_ID));

        // Some hard-coded prelude types are supported.
        let prelude = ctx.registry.get(&PRELUDE_ID).unwrap();
        match self.0.as_str() {
            "USize" => return prelude.get_type("usize"),
            "Q" => return prelude.get_type("qubit"),
            _ => {}
        }

        // Try to resolve the type in the current extension.
        if let Some(ty) = ext.get_type(&self.0) {
            return Some(ty);
        }

        // Try to resolve the type in the other extensions in scope.
        for ext in ctx.scope.iter() {
            if let Some(ty) = ctx.registry.get(ext).and_then(|ext| ext.get_type(&self.0)) {
                return Some(ty);
            }
        }

        None
    }
}
