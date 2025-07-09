//! Rows of types, used for function signatures,
//! designed to support efficient static allocation.

use std::{
    borrow::Cow,
    fmt::{self, Display, Write},
    ops::{Deref, DerefMut},
};

use super::{
    MaybeRV, NoRV, RowVariable, Substitution, Term, Transformable, Type, TypeArg, TypeBase, TypeRV,
    TypeTransformer, type_param::TypeParam,
};
use crate::{extension::SignatureError, utils::display_list};
use delegate::delegate;
use itertools::Itertools;

/// List of types, used for function signatures.
/// The `ROWVARS` parameter controls whether this may contain [`RowVariable`]s
#[derive(Clone, Eq, Debug, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
#[serde(transparent)]
pub struct TypeRowBase<ROWVARS: MaybeRV> {
    /// The datatypes in the row.
    types: Cow<'static, [TypeBase<ROWVARS>]>,
}

/// Row of single types i.e. of known length, for node inputs/outputs
pub type TypeRow = TypeRowBase<NoRV>;

/// Row of types and/or row variables, the number of actual types is thus
/// unknown
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
    #[must_use]
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
    #[must_use]
    pub fn as_slice(&self) -> &[TypeBase<RV>] {
        &self.types
    }

    /// Applies a substitution to the row.
    /// For `TypeRowRV`, note this may change the length of the row.
    /// For `TypeRow`, guaranteed not to change the length of the row.
    pub(crate) fn substitute(&self, s: &Substitution) -> Self {
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
            #[must_use] pub fn into_owned(self) -> Vec<TypeBase<RV>>;

            /// Returns `true` if the row contains no types.
            #[must_use] pub fn is_empty(&self) -> bool ;
        }
    }

    pub(super) fn validate(&self, var_decls: &[TypeParam]) -> Result<(), SignatureError> {
        self.iter().try_for_each(|t| t.validate(var_decls))
    }
}

impl<RV: MaybeRV> Transformable for TypeRowBase<RV> {
    fn transform<T: TypeTransformer>(&mut self, tr: &T) -> Result<bool, T::Err> {
        self.to_mut().transform(tr)
    }
}

impl TypeRow {
    delegate! {
        to self.types {
            /// Returns the number of types in the row.
            #[must_use] pub fn len(&self) -> usize;

            #[inline(always)]
            /// Returns the type at the specified index. Returns `None` if out of bounds.
            #[must_use] pub fn get(&self, offset: usize) -> Option<&Type>;
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
                .map(std::convert::TryInto::try_into)
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

// Fallibly convert a [Term] to a [TypeRV].
//
// This will fail if `arg` is of non-type kind (e.g. String).
impl TryFrom<Term> for TypeRV {
    type Error = SignatureError;

    fn try_from(value: Term) -> Result<Self, Self::Error> {
        match value {
            TypeArg::Runtime(ty) => Ok(ty.into()),
            TypeArg::Variable(v) => Ok(TypeRV::new_row_var_use(
                v.index(),
                v.bound_if_row_var()
                    .ok_or(SignatureError::InvalidTypeArgs)?,
            )),
            _ => Err(SignatureError::InvalidTypeArgs),
        }
    }
}

// Fallibly convert a [Term] to a [TypeRow].
//
// This will fail if `arg` is of non-sequence kind (e.g. Type)
// or if the sequence contains row variables.
impl TryFrom<Term> for TypeRow {
    type Error = SignatureError;

    fn try_from(value: TypeArg) -> Result<Self, Self::Error> {
        match value {
            TypeArg::List(elems) => elems
                .into_iter()
                .map(|ta| ta.as_runtime())
                .collect::<Option<Vec<_>>>()
                .map(|x| x.into())
                .ok_or(SignatureError::InvalidTypeArgs),
            _ => Err(SignatureError::InvalidTypeArgs),
        }
    }
}

// Fallibly convert a [TypeArg] to a [TypeRowRV].
//
// This will fail if `arg` is of non-sequence kind (e.g. Type).
impl TryFrom<Term> for TypeRowRV {
    type Error = SignatureError;

    fn try_from(value: Term) -> Result<Self, Self::Error> {
        match value {
            TypeArg::List(elems) => elems
                .into_iter()
                .map(TypeRV::try_from)
                .collect::<Result<Vec<_>, _>>()
                .map(|vec| vec.into()),
            TypeArg::Variable(v) => Ok(vec![TypeRV::new_row_var_use(
                v.index(),
                v.bound_if_row_var()
                    .ok_or(SignatureError::InvalidTypeArgs)?,
            )]
            .into()),
            _ => Err(SignatureError::InvalidTypeArgs),
        }
    }
}

impl From<TypeRow> for Term {
    fn from(value: TypeRow) -> Self {
        Term::List(value.into_owned().into_iter().map_into().collect())
    }
}

impl From<TypeRowRV> for Term {
    fn from(value: TypeRowRV) -> Self {
        Term::List(value.into_owned().into_iter().map_into().collect())
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
    use super::*;
    use crate::{
        extension::prelude::bool_t,
        types::{Type, TypeArg, TypeRV},
    };

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
                        .prop_map(|ts| ts.clone().into())
                        .boxed()
                }
            }
        }
    }

    #[test]
    fn test_try_from_term_to_typerv() {
        // Test successful conversion with Runtime type
        let runtime_type = Type::UNIT;
        let term = TypeArg::Runtime(runtime_type.clone());
        let result = TypeRV::try_from(term);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), TypeRV::from(runtime_type));

        // Test failure with non-type kind
        let term = Term::String("test".to_string());
        let result = TypeRV::try_from(term);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_from_term_to_typerow() {
        // Test successful conversion with List
        let types = vec![Type::new_unit_sum(1), bool_t()];
        let type_args = types.iter().map(|t| TypeArg::Runtime(t.clone())).collect();
        let term = TypeArg::List(type_args);
        let result = TypeRow::try_from(term);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), TypeRow::from(types));

        // Test failure with non-list
        let term = TypeArg::Runtime(Type::UNIT);
        let result = TypeRow::try_from(term);
        assert!(result.is_err());
    }

    #[test]
    fn test_try_from_term_to_typerowrv() {
        // Test successful conversion with List
        let types = [TypeRV::from(Type::UNIT), TypeRV::from(bool_t())];
        let type_args = types.iter().map(|t| t.clone().into()).collect();
        let term = TypeArg::List(type_args);
        let result = TypeRowRV::try_from(term);
        assert!(result.is_ok());

        // Test failure with non-sequence kind
        let term = Term::String("test".to_string());
        let result = TypeRowRV::try_from(term);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_typerow_to_term() {
        let types = vec![Type::UNIT, bool_t()];
        let type_row = TypeRow::from(types);
        let term = Term::from(type_row);

        match term {
            Term::List(elems) => {
                assert_eq!(elems.len(), 2);
            }
            _ => panic!("Expected Term::List"),
        }
    }

    #[test]
    fn test_from_typerowrv_to_term() {
        let types = vec![TypeRV::from(Type::UNIT), TypeRV::from(bool_t())];
        let type_row_rv = TypeRowRV::from(types);
        let term = Term::from(type_row_rv);

        match term {
            TypeArg::List(elems) => {
                assert_eq!(elems.len(), 2);
            }
            _ => panic!("Expected Term::List"),
        }
    }
}
