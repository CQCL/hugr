//! Rows of types, used for function signatures,
//! designed to support efficient static allocation.

use std::{
    borrow::Cow,
    fmt::{self, Display, Write},
    ops::{Deref, DerefMut},
};

use super::{type_param::TypeParam, MaybeRV, NoRV, RowVariable, Substitution, Type, TypeBase};
use crate::{
    extension::{ExtensionRegistry, SignatureError},
    utils::display_list,
};
use delegate::delegate;
use itertools::Itertools;

/// List of types, used for function signatures.
/// The `ROWVARS` parameter controls whether this may contain [RowVariable]s
#[derive(Clone, Eq, Debug, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(transparent)]
pub struct TypeRowBase<ROWVARS: MaybeRV> {
    /// The datatypes in the row.
    types: Cow<'static, [TypeBase<ROWVARS>]>,
}

/// Row of single types i.e. of known length, for node inputs/outputs
pub type TypeRow = TypeRowBase<NoRV>;

/// Row of types and/or row variables, the number of actual types is thus unknown
pub type TypeRowRV = TypeRowBase<RowVariable>;

impl<RV1: MaybeRV, RV2: MaybeRV> PartialEq<TypeRowBase<RV1>> for TypeRowBase<RV2> {
    fn eq(&self, other: &TypeRowBase<RV1>) -> bool {
        self.types.len() == other.types.len()
            && self
                .types
                .iter()
                .zip(other.types.iter())
                .all(|(s, o)| s == o)
    }
}

impl<RV: MaybeRV> Display for TypeRowBase<RV> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;
        display_list(self.types.as_ref(), f)?;
        f.write_char(']')
    }
}

impl<RV: MaybeRV> TypeRowBase<RV> {
    /// Create a new empty row.
    pub const fn new() -> Self {
        Self {
            types: Cow::Owned(Vec::new()),
        }
    }

    /// Returns a new `TypeRow` with `xs` concatenated onto `self`.
    pub fn extend<'a>(&'a self, rest: impl IntoIterator<Item = &'a TypeBase<RV>>) -> Self {
        self.iter().chain(rest).cloned().collect_vec().into()
    }

    /// Returns a reference to the types in the row.
    pub fn as_slice(&self) -> &[TypeBase<RV>] {
        &self.types
    }

    /// Applies a substitution to the row.
    /// For `TypeRowRV`, note this may change the length of the row.
    /// For `TypeRow`, guaranteed not to change the length of the row.
    pub(super) fn substitute(&self, s: &Substitution) -> Self {
        self.iter()
            .flat_map(|ty| ty.substitute(s))
            .collect::<Vec<_>>()
            .into()
    }

    delegate! {
        to self.types {
            /// Iterator over the types in the row.
            pub fn iter(&self) -> impl Iterator<Item = &TypeBase<RV>>;

            /// Mutable vector of the types in the row.
            pub fn to_mut(&mut self) -> &mut Vec<TypeBase<RV>>;

            /// Allow access (consumption) of the contained elements
            pub fn into_owned(self) -> Vec<TypeBase<RV>>;

            /// Returns `true` if the row contains no types.
            pub fn is_empty(&self) -> bool ;
        }
    }

    pub(super) fn validate(
        &self,
        exts: &ExtensionRegistry,
        var_decls: &[TypeParam],
    ) -> Result<(), SignatureError> {
        self.iter().try_for_each(|t| t.validate(exts, var_decls))
    }
}

impl TypeRow {
    delegate! {
        to self.types {
            /// Returns the number of types in the row.
            pub fn len(&self) -> usize;

            #[inline(always)]
            /// Returns the type at the specified index. Returns `None` if out of bounds.
            pub fn get(&self, offset: usize) -> Option<&Type>;
        }

        to self.types.to_mut() {
            #[inline(always)]
            /// Returns the type at the specified index. Returns `None` if out of bounds.
            pub fn get_mut(&mut self, offset: usize) -> Option<&mut Type>;
        }
    }
}

impl TryFrom<TypeRowRV> for TypeRow {
    type Error = SignatureError;

    fn try_from(value: TypeRowRV) -> Result<Self, Self::Error> {
        Ok(Self::from(
            value
                .into_owned()
                .into_iter()
                .map(|t| t.try_into())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|var| SignatureError::RowVarWhereTypeExpected { var })?,
        ))
    }
}

impl<RV: MaybeRV> Default for TypeRowBase<RV> {
    fn default() -> Self {
        Self::new()
    }
}

impl<RV: MaybeRV> From<Vec<TypeBase<RV>>> for TypeRowBase<RV> {
    fn from(types: Vec<TypeBase<RV>>) -> Self {
        Self {
            types: types.into(),
        }
    }
}

impl From<Vec<Type>> for TypeRowRV {
    fn from(types: Vec<Type>) -> Self {
        Self {
            types: types.into_iter().map(Type::into_).collect(),
        }
    }
}

impl From<TypeRow> for TypeRowRV {
    fn from(value: TypeRow) -> Self {
        Self {
            types: value.iter().cloned().map(Type::into_).collect(),
        }
    }
}

impl<RV: MaybeRV> From<&'static [TypeBase<RV>]> for TypeRowBase<RV> {
    fn from(types: &'static [TypeBase<RV>]) -> Self {
        Self {
            types: types.into(),
        }
    }
}

impl<RV1: MaybeRV> From<TypeBase<RV1>> for TypeRowRV {
    fn from(t: TypeBase<RV1>) -> Self {
        Self {
            types: vec![t.into_()].into(),
        }
    }
}

impl From<Type> for TypeRow {
    fn from(t: Type) -> Self {
        Self {
            types: vec![t].into(),
        }
    }
}

impl<RV: MaybeRV> Deref for TypeRowBase<RV> {
    type Target = [TypeBase<RV>];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<RV: MaybeRV> DerefMut for TypeRowBase<RV> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.types.to_mut()
    }
}

#[cfg(test)]
mod test {
    mod proptest {
        use crate::proptest::RecursionDepth;
        use crate::types::{MaybeRV, TypeBase, TypeRowBase};
        use ::proptest::prelude::*;

        impl<RV: MaybeRV> Arbitrary for super::super::TypeRowBase<RV> {
            type Parameters = RecursionDepth;
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(depth: Self::Parameters) -> Self::Strategy {
                use proptest::collection::vec;
                if depth.leaf() {
                    Just(TypeRowBase::new()).boxed()
                } else {
                    vec(any_with::<TypeBase<RV>>(depth), 0..4)
                        .prop_map(|ts| ts.to_vec().into())
                        .boxed()
                }
            }
        }
    }
}
