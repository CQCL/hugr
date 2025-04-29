//! Declarative operation definitions.
//!
//! This module defines a YAML schema for defining operations in a declarative way.
//!
//! See the [specification] and [`ExtensionSetDeclaration`] for more details.
//!
//! [specification]: https://github.com/CQCL/hugr/blob/main/specification/hugr.md#declarative-format
//! [`ExtensionSetDeclaration`]: super::ExtensionSetDeclaration

use std::collections::HashMap;
use std::sync::Weak;

use serde::{Deserialize, Serialize};
use smol_str::SmolStr;

use crate::Extension;
use crate::extension::{OpDef, SignatureFunc};
use crate::ops::OpName;
use crate::types::type_param::TypeParam;

use super::signature::SignatureDeclaration;
use super::{DeclarationContext, ExtensionDeclarationError};

/// A declarative operation definition.
///
/// TODO: The "Lowering" attribute is not yet supported.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct OperationDeclaration {
    /// The identifier the operation.
    name: OpName,
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
    misc: HashMap<String, serde_json::Value>,
    /// A pre-compiled lowering routine.
    ///
    /// This is not yet supported, and will raise an error if present.
    #[serde(default)]
    #[serde(skip_serializing_if = "crate::utils::is_default")]
    lowering: Option<String>,
}

impl OperationDeclaration {
    /// Register this operation in the given extension.
    ///
    /// Requires a [`Weak`] reference to the extension defining the operation.
    /// This method is intended to be used inside the closure passed to [`Extension::new_arc`].
    pub fn register<'ext>(
        &self,
        ext: &'ext mut Extension,
        ctx: DeclarationContext<'_>,
        extension_ref: &Weak<Extension>,
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

        if self.lowering.is_some() {
            return Err(ExtensionDeclarationError::LoweringNotSupported {
                ext: ext.name().clone(),
                op: self.name.clone(),
            });
        }

        let signature_func: SignatureFunc = signature.make_signature(ext, ctx, &params)?;

        let op_def = ext.add_op(
            self.name.clone(),
            self.description.clone(),
            signature_func,
            extension_ref,
        )?;

        for (k, v) in &self.misc {
            op_def.add_misc(k, v.clone());
        }

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
