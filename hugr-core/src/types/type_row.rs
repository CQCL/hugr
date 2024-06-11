//! Rows of types, used for function signatures,
//! designed to support efficient static allocation.

use std::{
    borrow::Cow,
    fmt::{self, Display, Write},
    ops::{Deref, DerefMut},
};

use super::{type_param::TypeParam, Implies, Substitution, Type};
use crate::{
    extension::{ExtensionRegistry, SignatureError},
    utils::display_list,
};
use delegate::delegate;
use itertools::Itertools;

/// List of types, used for function signatures.
#[derive(Clone, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(transparent)]
pub struct TypeRow<const ROWVARS: bool = false> {
    /// The datatypes in the row.
    types: Cow<'static, [Type<ROWVARS>]>,
}

impl<const RV1: bool, const RV2: bool> PartialEq<TypeRow<RV1>> for TypeRow<RV2> {
    fn eq(&self, other: &TypeRow<RV1>) -> bool {
        self.types.len() == other.types.len()
            && self
                .types
                .iter()
                .zip(other.types.iter())
                .all(|(s, o)| s == o)
    }
}

impl<const RV: bool> Display for TypeRow<RV> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;
        display_list(self.types.as_ref(), f)?;
        f.write_char(']')
    }
}

impl<const RV: bool> TypeRow<RV> {
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
        self.iter()
            .flat_map(|ty| ty.substitute(s))
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

    pub(super) fn validate(
        &self,
        exts: &ExtensionRegistry,
        var_decls: &[TypeParam],
    ) -> Result<(), SignatureError> {
        self.iter().try_for_each(|t| t.validate(exts, var_decls))
    }

    pub fn into_<const RV2: bool>(self) -> TypeRow<RV2> {
        let _ = Implies::<RV, RV2>::A_IMPLIES_B;
        TypeRow::from(
            self.types
                .iter()
                .cloned()
                .map(Type::into_)
                .collect::<Vec<_>>(),
        )
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

/*impl From<TypeRow> for TypeRow<true> {
    fn from(value: TypeRow) -> Self {
        Self::from(value.into_owned().into_iter().map_into().collect::<Vec<Type<true>>>())
    }
}*/

/*impl Into<Cow<'static, [Type<true>]>> for TypeRow<false> {
    fn into(self) -> Cow<'static, [Type<true>]> {
        let tr = self.types.into_iter().cloned().map(|t| t.into()).collect();
        Cow::Owned(tr)
    }
}*/

impl TryFrom<TypeRow<true>> for TypeRow {
    type Error = SignatureError;

    fn try_from(value: TypeRow<true>) -> Result<Self, Self::Error> {
        Ok(Self::from(
            value
                .into_owned()
                .into_iter()
                .map(|t| t.try_into())
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

impl<const RV: bool> Default for TypeRow<RV> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const RV: bool> From<Vec<Type<RV>>> for TypeRow<RV> {
    fn from(types: Vec<Type<RV>>) -> Self {
        Self {
            types: types.into()
        }
    }
}

impl From<Vec<Type>> for TypeRow<true> {
    fn from(types: Vec<Type>) -> Self {
        Self {
            types: types.into_iter().map(Type::into_).collect()
        }
    }
}

impl From<TypeRow> for TypeRow<true> {
    fn from(value: TypeRow) -> Self {
        Self {
            types: value.into_iter().cloned().map(Type::into_).collect()
        }
    }
}

impl<const RV: bool> From<&'static [Type<RV>]> for TypeRow<RV> {
    fn from(types: &'static [Type<RV>]) -> Self {
        Self {
            types: types.into()
        }
    }
}

impl<const RV1: bool> From<Type<RV1>> for TypeRow<true> {
    fn from(t: Type<RV1>) -> Self {
        Self {
            types: vec![t.into_()].into(),
        }
    }
}

impl From<Type> for TypeRow {
    fn from(t: Type) -> Self {
        Self {
            types: vec![t].into()
        }
    }
}

impl<const RV: bool> Deref for TypeRow<RV> {
    type Target = [Type<RV>];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<const RV: bool> DerefMut for TypeRow<RV> {
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
