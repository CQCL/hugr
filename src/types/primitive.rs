//! Primitive types which are leaves of the type tree

use crate::ops::AliasDecl;

use super::{type_param::TypeParam, CustomType, PolyFuncType, Type, TypeArg, TypeBound};

/// Index of a type variable.
/// Roughly DeBruijn, but note that when many variables are declared by the same binder ([PolyFuncType]),
/// index 0 ("the closest binder") refers to the first element of the array of binders, i.e. as if the
/// binders were processed in reverse order (highest-index element first);
/// whereas [PolyFuncType::instantiate] instantiates lowest-index elements first (and leaves highest-index
/// elements uninstantiated if not enough values are provided for every binder).
#[derive(
    Clone,
    Copy,
    Debug,
    derive_more::Display,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
#[display(fmt = "<{}>", _0)]
pub struct VarIdx(usize);

impl VarIdx {
    /// Constructor, given DeBruijn index
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    pub(super) fn in_outer_scope(&self, num_binders: usize) -> Option<Self> {
        self.0.checked_sub(num_binders).map(VarIdx)
    }

    pub(super) fn as_typearg(&self, decl: TypeParam) -> TypeArg {
        TypeArg::new_var_use(self.0, decl)
    }

    #[allow(unused)]
    pub(super) fn as_type(&self, bound: TypeBound) -> Type {
        Type::new_var_use(self.0, bound)
    }
}

impl From<VarIdx> for usize {
    fn from(value: VarIdx) -> Self {
        value.0
    }
}

impl std::ops::Add<usize> for VarIdx {
    type Output = VarIdx;

    fn add(self, rhs: usize) -> Self {
        Self(self.0 + rhs)
    }
}

#[derive(
    Clone, PartialEq, Debug, Eq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
/// Representation of a Primitive type, i.e. neither a Sum nor a Tuple.
pub enum PrimType {
    // TODO optimise with Box<CustomType> ?
    // or some static version of this?
    #[allow(missing_docs)]
    Extension(CustomType),
    #[allow(missing_docs)]
    #[display(fmt = "Alias({})", "_0.name()")]
    Alias(AliasDecl),
    #[allow(missing_docs)]
    #[display(fmt = "Function({})", "_0")]
    Function(Box<PolyFuncType>),
    // DeBruijn index, and cache of TypeBound (checked in validation)
    #[allow(missing_docs)]
    #[display(fmt = "Variable({})", _0)]
    Variable(VarIdx, TypeBound),
}

impl PrimType {
    /// Returns the bound of this [`PrimType`].
    pub fn bound(&self) -> TypeBound {
        match self {
            PrimType::Extension(c) => c.bound(),
            PrimType::Alias(a) => a.bound,
            PrimType::Function(_) => TypeBound::Copyable,
            PrimType::Variable(_, b) => *b,
        }
    }
}
