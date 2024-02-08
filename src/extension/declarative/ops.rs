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

use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::extension::{ExtensionRegistry, ExtensionSet, OpDef, SignatureFunc};
use crate::types::type_param::TypeParam;
use crate::Extension;

use super::signature::SignatureDeclaration;
use super::ExtensionDeclarationError;

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
    pub fn register<'ext>(
        &self,
        ext: &'ext mut Extension,
        scope: &ExtensionSet,
        registry: &ExtensionRegistry,
    ) -> Result<&'ext mut OpDef, ExtensionDeclarationError> {
        // We currently only support explicit signatures.
        //
        // TODO: Support missing signatures?
        let Some(signature) = &self.signature else {
            return Err(ExtensionDeclarationError::MissingSignature {
                ext: ext.name().clone(),
                op: self.name.clone(),
            });
        };

        // We currently do not support parametric operations.
        if !self.params.is_empty() {
            return Err(ExtensionDeclarationError::ParametricOperation {
                ext: ext.name().clone(),
                op: self.name.clone(),
            });
        }
        let params: Vec<TypeParam> = vec![];

        let signature_func: SignatureFunc =
            signature.make_signature(ext, scope, registry, &params)?;

        let op_def = ext.add_op(self.name.clone(), self.description.clone(), signature_func)?;
        Ok(op_def)
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
pub(super) struct ParamDeclaration(
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
