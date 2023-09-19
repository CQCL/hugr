//! Primitive types which are leaves of the type tree

use itertools::Itertools;

use crate::{
    extension::{ExtensionRegistry, SignatureError},
    ops::AliasDecl,
};

use super::{type_param::TypeParam, CustomType, FunctionType, Substitution, TypeBound};

/// A polymorphic function type, e.g. of a [Graph], or perhaps an [OpDef].
/// (Nodes/operations in the Hugr are not polymorphic.)
///
/// [Graph]: crate::values::PrimValue::Function
/// [OpDef]: crate::extension::OpDef
#[derive(
    Clone, PartialEq, Debug, Eq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
#[display(
    fmt = "forall {}. {}",
    "params.iter().map(ToString::to_string).join(\" \")",
    "body"
)]
pub struct PolyFuncType {
    /// The declared type parameters, i.e., these must be instantiated with
    /// the same number of [TypeArg]s before the function can be called.
    ///
    /// [TypeArg]: super::type_param::TypeArg
    pub params: Vec<TypeParam>,
    /// Template for the function. May contain variables up to length of [Self::params]
    pub body: Box<FunctionType>,
}

impl From<FunctionType> for PolyFuncType {
    fn from(body: FunctionType) -> Self {
        Self {
            params: vec![],
            body: Box::new(body),
        }
    }
}

impl PolyFuncType {
    pub(super) fn validate(
        &self,
        reg: &ExtensionRegistry,
        type_vars: &[TypeParam],
    ) -> Result<(), SignatureError> {
        // TODO should we add a mechanism to validate a TypeParam?
        let mut v = vec![];
        let all_type_vars = if self.params.is_empty() {
            type_vars
        } else {
            // Type vars declared here go at lowest indices (as per DeBruijn)
            v.extend(type_vars.iter().cloned());
            v.extend_from_slice(type_vars);
            v.as_slice()
        };
        self.body.validate(reg, all_type_vars)
    }

    pub(super) fn substitute(&self, exts: &ExtensionRegistry, sub: &Substitution) -> Self {
        Self {
            body: Box::new(
                self.body
                    .substitute(exts, &sub.enter_scope(self.params.len())),
            ),
            params: self.params.clone(),
        }
    }
}

#[derive(
    Clone, PartialEq, Debug, Eq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
pub(super) enum PrimType {
    // TODO optimise with Box<CustomType> ?
    // or some static version of this?
    Extension(CustomType),
    #[display(fmt = "Alias({})", "_0.name()")]
    Alias(AliasDecl),
    #[display(fmt = "Function({})", "_0")]
    Function(PolyFuncType),
    // DeBruijn index, and cache of TypeBound (checked in validation)
    #[display(fmt = "Variable({})", _0)]
    Variable(usize, TypeBound),
}

impl PrimType {
    pub(super) fn bound(&self) -> TypeBound {
        match self {
            PrimType::Extension(c) => c.bound(),
            PrimType::Alias(a) => a.bound,
            PrimType::Function(_) => TypeBound::Copyable,
            PrimType::Variable(_, b) => *b,
        }
    }
}
