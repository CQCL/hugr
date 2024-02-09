//! Declarative signature definitions.
//!
//! This module defines a YAML schema for defining the signature of an operation in a declarative way.
//!
//! See the [specification] and [`ExtensionSetDeclaration`] for more details.
//!
//! [specification]: https://github.com/CQCL/hugr/blob/main/specification/hugr.md#declarative-format
//! [`ExtensionSetDeclaration`]: super::ExtensionSetDeclaration

use itertools::Itertools;
use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::extension::prelude::PRELUDE_ID;
use crate::extension::{
    CustomValidator, ExtensionRegistry, ExtensionSet, SignatureFunc, TypeDef, TypeParametrised,
};
use crate::types::type_param::TypeParam;
use crate::types::{CustomType, FunctionType, PolyFuncType, Type, TypeRow};
use crate::utils::is_default;
use crate::Extension;

use super::ExtensionDeclarationError;

/// A declarative operation signature definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct SignatureDeclaration {
    /// The inputs to the operation.
    inputs: Vec<SignaturePortDeclaration>,
    /// The outputs of the operation.
    outputs: Vec<SignaturePortDeclaration>,
    /// A set of extensions invoked while running this operation.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    extensions: ExtensionSet,
}

impl SignatureDeclaration {
    pub fn make_signature(
        &self,
        ext: &Extension,
        scope: &ExtensionSet,
        registry: &ExtensionRegistry,
        op_params: &[TypeParam],
    ) -> Result<SignatureFunc, ExtensionDeclarationError> {
        let body = FunctionType {
            input: self.make_type_row(&self.inputs, ext, scope, registry, op_params)?,
            output: self.make_type_row(&self.outputs, ext, scope, registry, op_params)?,
            extension_reqs: self.extensions.clone(),
        };

        let poly_func = PolyFuncType::new(op_params, body);
        Ok(SignatureFunc::TypeScheme(CustomValidator::from_polyfunc(
            poly_func,
        )))
    }

    /// Create a type row from a list of port declarations.
    fn make_type_row(
        &self,
        v: &[SignaturePortDeclaration],
        ext: &Extension,
        scope: &ExtensionSet,
        registry: &ExtensionRegistry,
        op_params: &[TypeParam],
    ) -> Result<TypeRow, ExtensionDeclarationError> {
        let types = v
            .iter()
            .map(|port_decl| port_decl.make_types(ext, scope, registry, op_params))
            .flatten_ok()
            .collect::<Result<Vec<Type>, _>>()?;
        Ok(types.into())
    }
}

/// A declarative definition for a number of ports in a signature's input or output.
///
/// Serialized as a 2 or 3-element list, where:
/// - The first element is an optional human-readable name of the port.
/// - The second element is the port type id.
/// - The optional third element is either:
///     - A number, indicating a repetition of the port that amount of times.
///     - A parameter identifier, to use as the repetition number.
///     - Nothing, in which case we default to a single repetition.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct SignaturePortDeclaration {
    /// The description of the ports. May be `null`.
    description: Option<String>,
    /// The type for the port.
    type_decl: TypeDeclaration,
    /// The number of repetitions for this port definition.
    repeat: PortRepetitionDeclaration,
}

impl SignaturePortDeclaration {
    /// Return an iterator with the types for this port declaration.
    ///
    /// Only a fixed number of repetitions is supported for now.
    ///
    /// TODO: We may need to use custom signature functions if `repeat` depends on a variable.
    fn make_types(
        &self,
        ext: &Extension,
        scope: &ExtensionSet,
        registry: &ExtensionRegistry,
        op_params: &[TypeParam],
    ) -> Result<impl Iterator<Item = Type>, ExtensionDeclarationError> {
        let n: usize = match &self.repeat {
            PortRepetitionDeclaration::Count(n) => *n,
            PortRepetitionDeclaration::Parameter(parametric_repetition) => {
                return Err(ExtensionDeclarationError::UnsupportedPortRepetition {
                    ext: ext.name().clone(),
                    parametric_repetition: parametric_repetition.clone(),
                })
            }
        };

        let ty = self.type_decl.make_type(ext, scope, registry, op_params)?;
        let ty = Type::new_extension(ty);

        Ok(itertools::repeat_n(ty, n))
    }
}

impl Serialize for SignaturePortDeclaration {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let len = if is_default(&self.repeat) { 2 } else { 3 };
        let mut seq = serializer.serialize_seq(Some(len))?;
        seq.serialize_element(&self.description)?;
        seq.serialize_element(&self.type_decl)?;
        if !is_default(&self.repeat) {
            seq.serialize_element(&self.repeat)?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for SignaturePortDeclaration {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct TypeParamVisitor;
        const EXPECTED_MSG: &str = "a 2-element list containing a type parameter name and id";

        impl<'de> serde::de::Visitor<'de> for TypeParamVisitor {
            type Value = SignaturePortDeclaration;

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
                let type_decl = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &EXPECTED_MSG))?;
                let repeat = seq.next_element()?.unwrap_or_default();
                Ok(SignaturePortDeclaration {
                    description,
                    type_decl,
                    repeat,
                })
            }
        }

        deserializer.deserialize_seq(TypeParamVisitor)
    }
}

/// A number of repetitions for a signature's port definition.
///
/// This value may be either:
/// - A number, indicating a repetition of the port that amount of times.
/// - A parameter identifier, to use as the repetition number.
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
        scope: &ExtensionSet,
        registry: &ExtensionRegistry,
        _op_params: &[TypeParam],
    ) -> Result<CustomType, ExtensionDeclarationError> {
        // The prelude is always in scope.
        debug_assert!(scope.contains(&PRELUDE_ID));

        // Only hard-coded prelude types are supported for now.
        let prelude = registry.get(&PRELUDE_ID).unwrap();
        let op_def: &TypeDef = match self.0.as_str() {
            "USize" => prelude.get_type("usize"),
            "Q" => prelude.get_type("qubit"),
            _ => {
                return Err(ExtensionDeclarationError::UnknownType {
                    ext: ext.name().clone(),
                    ty: self.0.clone(),
                })
            }
        }
        .ok_or(ExtensionDeclarationError::UnknownType {
            ext: ext.name().clone(),
            ty: self.0.clone(),
        })?;

        // The hard-coded types are not parametric.
        assert!(op_def.params().is_empty());
        let op = op_def.instantiate(&[]).unwrap();

        Ok(op)
    }
}
