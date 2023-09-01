//! Polymorphic type schemes for ops.
//! The type scheme declares a number of TypeParams; any TypeArgs fitting those,
//! produce a FunctionType for the Op by substitution.

use crate::types::type_param::{check_type_args, TypeArg, TypeParam};
use crate::types::FunctionType;

use super::{CustomSignatureFunc, ExtensionRegistry, SignatureError};

/// A polymorphic type scheme for an op
pub struct OpDefTypeScheme {
    /// The declared type parameters, i.e., every Op must provide [TypeArg]s for these
    pub params: Vec<TypeParam>,
    /// Template for the Op type. May contain variables up to length of [OpDefTypeScheme::params]
    body: FunctionType,
}

impl OpDefTypeScheme {
    /// Create a new OpDefTypeScheme.
    ///
    /// #Errors
    /// Validates that all types in the schema are well-formed and all variables in the body
    /// are declared with [TypeParam]s that guarantee they will fit.
    pub fn new(
        params: impl Into<Vec<TypeParam>>,
        body: FunctionType,
        extension_registry: &ExtensionRegistry,
    ) -> Result<Self, SignatureError> {
        let params = params.into();
        body.validate(extension_registry, &params)?;
        Ok(Self { params, body })
    }
}

impl CustomSignatureFunc for OpDefTypeScheme {
    fn compute_signature(
        &self,
        _name: &smol_str::SmolStr,
        args: &[TypeArg],
        _misc: &std::collections::HashMap<String, serde_yaml::Value>,
    ) -> Result<FunctionType, SignatureError> {
        check_type_args(args, &self.params).map_err(SignatureError::TypeArgMismatch)?;
        Ok(self.body.substitute(args))
    }
}
