//! Declarative operation definitions.
//!
//! This module defines a YAML schema for defining operations in a declarative way.
//!
//! See the [specification] and [`ExtensionSetDeclaration`] for more details.
//!
//! [specification]: https://github.com/CQCL/hugr/blob/main/specification/hugr.md#declarative-format
//! [`ExtensionSetDeclaration`]: super::ExtensionSetDeclaration

use std::collections::HashMap;
use std::path::PathBuf;

use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::extension::{
    ExtensionBuildError, ExtensionRegistry, ExtensionSet, OpDef, SignatureFunc,
};
use crate::types::TypeName;
use crate::utils::is_default;
use crate::Extension;

/// A declarative operation definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct OperationDeclaration {
    /// The identifier the operation.
    name: SmolStr,
    /// A description for the operation.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    description: String,
    /// The signature of the operation.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    signature: Option<SignatureDeclaration>,
    /// A set of per-node parameters required to instantiate this operation.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    params: HashMap<SmolStr, ParamDeclaration>,
    /// An extra set of data associated to the operation.
    ///
    /// This data is kept in the Hugr, and may be accessed by the relevant runtime.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    misc: HashMap<SmolStr, serde_yaml::Value>,
    /// A pre-compiled lowering routine.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    lowering: Option<LoweringDeclaration>,
}

impl OperationDeclaration {
    /// Register this operation in the given extension.
    #[allow(unused, unreachable_code, clippy::diverging_sub_expression)]
    pub fn register<'ext>(
        &self,
        ext: &'ext mut Extension,
        scope: &ExtensionSet,
        registry: &ExtensionRegistry,
    ) -> Result<&'ext mut OpDef, ExtensionBuildError> {
        let signature_func: SignatureFunc = unimplemented!("signature_func");
        ext.add_op(self.name.clone(), self.description.clone(), signature_func)
    }
}

/// A declarative operation signature definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct SignatureDeclaration {
    /// The inputs to the operation.
    inputs: Vec<SignaturePortDeclaration>,
    /// The outputs of the operation.
    outputs: Vec<SignaturePortDeclaration>,
    /// A set of extensions invoked while running this operation.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    extensions: ExtensionSet,
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
    /// The type identifier for the port.
    ///
    /// TODO: The spec definition is more complex than just a type identifier,
    /// we should be able to support expressions like:
    ///
    /// - `Q`
    /// - `Array<i>(Array<j>(F64))`
    /// - `Function[r](USize -> USize)`
    /// - `Opaque(complex_matrix,i,j)`
    type_name: TypeName,
    /// The number of repetitions for this port definition.
    repeat: PortRepetitionDeclaration,
}

impl Serialize for SignaturePortDeclaration {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let len = if is_default(&self.repeat) { 2 } else { 3 };
        let mut seq = serializer.serialize_seq(Some(len))?;
        seq.serialize_element(&self.description)?;
        seq.serialize_element(&self.type_name)?;
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
                let type_name = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &EXPECTED_MSG))?;
                let repeat = seq.next_element()?.unwrap_or_default();
                Ok(SignaturePortDeclaration {
                    description,
                    type_name,
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

/// The type of a per-node operation parameter required to instantiate an operation.
///
/// TODO: The value should be decoded as a [`TypeParam`].
/// Valid options include:
///
/// - `USize`
/// - `Type`
///
/// [`TypeParam`]: crate::types::type_param::TypeParam
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
struct ParamDeclaration(
    /// TODO: Store a [`TypeParam`], and implement custom parsers.
    ///
    /// [`TypeParam`]: crate::types::type_param::TypeParam
    String,
);

/// Reference to a binary lowering function.
///
/// TODO: How this works is not defined in the spec. This is currently a stub.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
struct LoweringDeclaration {
    /// Path to the lowering executable.
    file: PathBuf,
    /// A set of extensions invoked while running this operation.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    extensions: ExtensionSet,
}
