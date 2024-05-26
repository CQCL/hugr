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
pub struct TypeRow<const ROWVARS:bool = false> {
    /// The datatypes in the row.
    types: Cow<'static, [Type<ROWVARS>]>,
}

impl <const RV:bool> Display for TypeRow<RV> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;
        display_list(self.types.as_ref(), f)?;
        f.write_char(']')
    }
}

impl <const RV:bool> TypeRow<RV> {
    /// Create a new empty row.
    pub const fn new() -> Self {
        Self {
            types: Cow::Owned(Vec::new()),
        }
    }

    /// Returns a new `TypeRow` with `xs` concatenated onto `self`.
    pub fn extend<'a>(&'a self, rest: impl IntoIterator<Item = &'a Type<RV>>) -> Self {
        self.iter().chain(rest).cloned().collect_vec().into()
    }

    /// Returns a reference to the types in the row.
    pub fn as_slice(&self) -> &[Type<RV>] {
        &self.types
    }

    /// Applies a substitution to the row.
    /// For `TypeRow<true>`, note this may change the length of the row.
    /// For `TypeRow<false>`, guaranteed not to change the length of the row.
    pub(super) fn substitute(&self, s: &Substitution) -> Self {
        self
            .iter()
            .flat_map(|ty| ty.subst_vec(s))
            .collect::<Vec<_>>()
            .into()
    }

    delegate! {
        to self.types {
            /// Iterator over the types in the row.
            pub fn iter(&self) -> impl Iterator<Item = &Type<RV>>;

            /// Mutable vector of the types in the row.
            pub fn to_mut(&mut self) -> &mut Vec<Type<RV>>;

            /// Allow access (consumption) of the contained elements
            pub fn into_owned(self) -> Vec<Type<RV>>;

            /// Returns `true` if the row contains no types.
            pub fn is_empty(&self) -> bool ;
        }
    }
}

impl TypeRow<false> {
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

impl TypeRow<true> {
    pub(super) fn validate(
        &self,
        exts: &ExtensionRegistry,
        var_decls: &[TypeParam],
    ) -> Result<(), SignatureError> {
        self.iter()
            .try_for_each(|t| t.validate(exts, var_decls))
    }
}

impl From<TypeRow> for TypeRow<true> {
    fn from(value: TypeRow) -> Self {
        Self::from(value.into_owned().into_iter().map_into().collect::<Vec<Type<true>>>())
    }
}

impl TryFrom<TypeRow<true>> for TypeRow {
    type Error = SignatureError;

    fn try_from(value: TypeRow<true>) -> Result<Self, Self::Error> {
        Ok(Self::from(value.into_owned().into_iter().map(|t| t.try_into()).collect::<Result<Vec<_>,_>>()?))
    }
}

impl <const RV:bool> Default for TypeRow<RV> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F, const RV:bool> From<F> for TypeRow<RV>
where
    F: Into<Cow<'static, [Type<RV>]>>,
{
    fn from(types: F) -> Self {
        Self {
            types: types.into(),
        }
    }
}

impl <const RV:bool> From<Type<RV>> for TypeRow<RV> {
    fn from(t: Type<RV>) -> Self {
        Self {
            types: vec![t].into(),
        }
    }
}

impl <const RV:bool> Deref for TypeRow<RV> {
    type Target = [Type<RV>];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl <const RV:bool> DerefMut for TypeRow<RV> {
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
