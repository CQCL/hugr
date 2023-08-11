#![allow(missing_docs)]

use std::fmt::Write;

use crate::{ops::AliasDecl, utils::display_list};
use std::fmt::{self, Debug, Display};

use super::{
    leaf::{least_upper_bound, PrimType, TypeBound},
    type_param::TypeArg,
    AbstractSignature, CustomType,
};

mod serialize;

#[derive(Clone, PartialEq, Debug, Eq, derive_more::Display)]
enum TypeEnum {
    Prim(PrimType),
    #[display(fmt = "Array[{};{}]", "_0", "_1")]
    Array(Box<Type>, usize),
    #[display(fmt = "Tuple({})", "DisplayRow(_0)")]
    Tuple(Vec<Type>),
    #[display(fmt = "Sum({})", "DisplayRow(_0)")]
    Sum(Vec<Type>),
}
impl TypeEnum {
    fn least_upper_bound(&self) -> Option<TypeBound> {
        match self {
            TypeEnum::Prim(p) => p.bound(),
            TypeEnum::Array(t, _) => t.least_upper_bound(),
            TypeEnum::Tuple(ts) => least_upper_bound(ts.iter().map(Type::least_upper_bound)),
            TypeEnum::Sum(ts) => least_upper_bound(ts.iter().map(Type::least_upper_bound)),
        }
    }
}

#[derive(
    Clone, PartialEq, Debug, Eq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
#[display(fmt = "{}", "_0")]
#[serde(into = "serialize::SerSimpleType", from = "serialize::SerSimpleType")]
pub struct Type(TypeEnum, Option<TypeBound>);

struct DisplayRow<'a>(&'a Vec<Type>);
impl<'a> Display for DisplayRow<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;
        display_list(self.0, f)?;
        f.write_char(']')
    }
}

impl Type {
    pub fn graph(_signature: AbstractSignature) -> Self {
        Self::new_extension(
            crate::resource::PRELUDE
                .get_type("graph")
                .unwrap()
                .instantiate_concrete(vec![
                    TypeArg::Sequence(
                        _signature
                            .input
                            .iter()
                            .cloned()
                            .map(TypeArg::Type)
                            .collect(),
                    ),
                    TypeArg::Sequence(
                        _signature
                            .output
                            .iter()
                            .cloned()
                            .map(TypeArg::Type)
                            .collect(),
                    ),
                ])
                .unwrap(),
        )
    }

    pub fn usize() -> Self {
        Self::new_extension(
            crate::resource::PRELUDE
                .get_type("usize")
                .unwrap()
                .instantiate_concrete(vec![])
                .unwrap(),
        )
    }
    pub fn new_tuple(types: impl IntoIterator<Item = Type>) -> Self {
        Self::new(TypeEnum::Tuple(types.into_iter().collect()))
    }

    /// New unit type, defined as an empty Tuple.
    pub fn new_unit() -> Self {
        Self::new_tuple(vec![])
    }

    pub fn new_sum(types: impl IntoIterator<Item = Type>) -> Self {
        Self::new(TypeEnum::Sum(types.into_iter().collect()))
    }

    pub fn new_extension(opaque: CustomType) -> Self {
        Self::new(TypeEnum::Prim(PrimType::E(opaque)))
    }
    pub fn new_alias(alias: AliasDecl) -> Self {
        Self::new(TypeEnum::Prim(PrimType::A(alias)))
    }

    fn new(type_e: TypeEnum) -> Self {
        let bound = type_e.least_upper_bound();
        Self(type_e, bound)
    }

    pub fn new_array(typ: Type, size: usize) -> Self {
        Self::new(TypeEnum::Array(Box::new(typ), size))
    }
    /// New Sum of Tuple types, used as predicates in branching.
    /// Tuple rows are defined in order by input rows.
    pub fn new_predicate<V>(variant_rows: impl IntoIterator<Item = V>) -> Self
    where
        V: IntoIterator<Item = Type>,
    {
        Self::new_sum(predicate_variants_row(variant_rows))
    }

    /// New simple predicate with empty Tuple variants
    pub fn new_simple_predicate(size: usize) -> Self {
        Self::new_predicate(std::iter::repeat(vec![]).take(size))
    }

    pub fn least_upper_bound(&self) -> Option<TypeBound> {
        self.1
    }
}

/// Return the type row of variants required to define a Sum of Tuples type
/// given the rows of each tuple
pub(crate) fn predicate_variants_row<V>(variant_rows: impl IntoIterator<Item = V>) -> Vec<Type>
where
    V: IntoIterator<Item = Type>,
{
    variant_rows.into_iter().map(Type::new_tuple).collect()
}
