//! Polymorphic type schemes for ops.
//! The type scheme declares a number of TypeParams; any TypeArgs fitting those,
//! produce a FunctionType for the Op by substitution.

use crate::types::type_param::{check_type_args, TypeArg, TypeParam};
use crate::types::FunctionType;

use super::{CustomSignatureFunc, ExtensionRegistry, SignatureError};

/// A polymorphic type scheme for an op
pub struct OpDefTypeScheme<'a> {
    /// The declared type parameters, i.e., every Op must provide [TypeArg]s for these
    pub params: Vec<TypeParam>,
    /// Template for the Op type. May contain variables up to length of [OpDefTypeScheme::params]
    body: FunctionType,
    /// Extensions - the [TypeDefBound]s in here will be needed when we instantiate the [OpDefTypeScheme]
    /// into a [FunctionType].
    ///
    /// [TypeDefBound]: super::type_def::TypeDefBound
    // Note that if the lifetimes, etc., become too painful to store this reference in here,
    // and we'd rather own the necessary data, we really only need the TypeDefBounds not the other parts,
    // and the validation traversal in new() discovers the small subset of TypeDefBounds that
    // each OpDefTypeScheme actually needs.
    exts: &'a ExtensionRegistry,
}

impl<'a> OpDefTypeScheme<'a> {
    /// Create a new OpDefTypeScheme.
    ///
    /// #Errors
    /// Validates that all types in the schema are well-formed and all variables in the body
    /// are declared with [TypeParam]s that guarantee they will fit.
    pub fn new(
        params: impl Into<Vec<TypeParam>>,
        body: FunctionType,
        extension_registry: &'a ExtensionRegistry,
    ) -> Result<Self, SignatureError> {
        let params = params.into();
        body.validate(extension_registry, &params)?;
        Ok(Self {
            params,
            body,
            exts: extension_registry,
        })
    }
}

impl<'a> CustomSignatureFunc for OpDefTypeScheme<'a> {
    fn compute_signature(
        &self,
        _name: &smol_str::SmolStr,
        args: &[TypeArg],
        _misc: &std::collections::HashMap<String, serde_yaml::Value>,
    ) -> Result<FunctionType, SignatureError> {
        check_type_args(args, &self.params).map_err(SignatureError::TypeArgMismatch)?;
        Ok(self.body.substitute(self.exts, args))
    }
}
