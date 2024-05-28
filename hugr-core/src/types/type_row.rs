//! Rows of types, used for function signatures,
//! designed to support efficient static allocation.

use std::{
    borrow::Cow,
    fmt::{self, Display, Write},
    ops::{Deref, DerefMut},
};

use super::{type_param::TypeParam, Substitution, Type};
use crate::{
    extension::{ExtensionRegistry, SignatureError},
    utils::display_list,
};
use delegate::delegate;
use itertools::Itertools;

/// List of types, used for function signatures.
#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(transparent)]
pub struct TypeRow {
    /// The datatypes in the row.
    types: Cow<'static, [Type]>,
}

impl Display for TypeRow {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;
        display_list(self.types.as_ref(), f)?;
        f.write_char(']')
    }
}

impl TypeRow {
    /// Create a new empty row.
    pub const fn new() -> Self {
        Self {
            types: Cow::Owned(Vec::new()),
        }
    }

    /// Returns a new `TypeRow` with `xs` concatenated onto `self`.
    pub fn extend<'a>(&'a self, rest: impl IntoIterator<Item = &'a Type>) -> Self {
        self.iter().chain(rest).cloned().collect_vec().into()
    }

    /// Returns a reference to the types in the row.
    pub fn as_slice(&self) -> &[Type] {
        &self.types
    }

    delegate! {
        to self.types {
            /// Iterator over the types in the row.
            pub fn iter(&self) -> impl Iterator<Item = &Type>;

            /// Returns the number of types in the row.
            pub fn len(&self) -> usize;

            /// Mutable vector of the types in the row.
            pub fn to_mut(&mut self) -> &mut Vec<Type>;

            /// Allow access (consumption) of the contained elements
            pub fn into_owned(self) -> Vec<Type>;

            /// Returns `true` if the row contains no types.
            pub fn is_empty(&self) -> bool ;

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

    /// Applies a substitution to the row. Note this may change the length
    /// if-and-only-if the row contains any [RowVariable]s.
    ///
    /// [RowVariable]: [crate::types::TypeEnum::RowVariable]
    pub(super) fn substitute(&self, tr: &Substitution) -> TypeRow {
        let res = self
            .iter()
            .flat_map(|ty| ty.substitute(tr))
            .collect::<Vec<_>>()
            .into();
        res
    }

    pub(super) fn validate_var_len(
        &self,
        exts: &ExtensionRegistry,
        var_decls: &[TypeParam],
    ) -> Result<(), SignatureError> {
        self.iter()
            .try_for_each(|t| t.validate(true, exts, var_decls))
    }
}

impl Default for TypeRow {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> From<F> for TypeRow
where
    F: Into<Cow<'static, [Type]>>,
{
    fn from(types: F) -> Self {
        Self {
            types: types.into(),
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

impl Deref for TypeRow {
    type Target = [Type];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for TypeRow {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.types.to_mut()
    }
}

#[cfg(test)]
mod test {
    mod proptest {
        use crate::proptest::RecursionDepth;
        use crate::{type_row, types::Type};
        use ::proptest::prelude::*;

        impl Arbitrary for super::super::TypeRow {
            type Parameters = RecursionDepth;
            type Strategy = BoxedStrategy<Self>;
            fn arbitrary_with(depth: Self::Parameters) -> Self::Strategy {
                use proptest::collection::vec;
                if depth.leaf() {
                    Just(type_row![]).boxed()
                } else {
                    vec(any_with::<Type>(depth), 0..4)
                        .prop_map(|ts| ts.to_vec().into())
                        .boxed()
                }
            }
        }
    }
}
