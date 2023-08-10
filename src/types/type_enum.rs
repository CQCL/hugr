#![allow(missing_docs)]

use std::fmt::Write;

use smol_str::SmolStr;

use crate::{ops::AliasDecl, utils::display_list};
use std::fmt::{self, Debug, Display};

use super::{
    // leaf::{AnyLeaf, CopyableLeaf, EqLeaf, InvalidBound, Tagged, TypeClass},
    leaf::{containing_tag, PrimType, TypeTag},
    AbstractSignature,
    CustomType,
    // TypeTag,
};

mod serialize;
pub const USIZE: Type = Type::Prim(PrimType::E(CustomType::new_simple(
    SmolStr::new_inline("prelude"),
    SmolStr::new_inline("usize"),
    super::TypeTag::Hashable,
)));

pub const F64: Type = Type::Prim(PrimType::E(CustomType::new_simple(
    SmolStr::new_inline("prelude"),
    SmolStr::new_inline("f64"),
    super::TypeTag::Classic,
)));

#[derive(
    Clone, PartialEq, Debug, Eq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
#[display(fmt = "{}")]
#[serde(
    into = "serialize::SerSimpleType",
    try_from = "serialize::SerSimpleType"
)]
pub enum Type {
    Prim(PrimType),
    #[display(fmt = "Array[{};{}]", "_0", "_1")]
    Array(Box<Type>, usize),
    #[display(fmt = "Tuple({})", "DisplayRow(_0)")]
    Tuple(Vec<Type>),
    #[display(fmt = "Sum({})", "DisplayRow(_0)")]
    Sum(Vec<Type>),
}

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
        todo!()
    }
    pub const fn usize() -> Self {
        USIZE
    }
    pub fn new_tuple(types: impl IntoIterator<Item = Type>) -> Self {
        Self::Tuple(types.into_iter().collect())
    }

    /// New unit type, defined as an empty Tuple.
    pub fn new_unit() -> Self {
        Self::new_tuple(vec![])
    }

    pub fn new_sum(types: impl IntoIterator<Item = Type>) -> Self {
        Self::Sum(types.into_iter().collect())
    }

    pub fn new_extension(opaque: CustomType) -> Self {
        Self::Prim(PrimType::E(opaque))
    }
    pub fn new_alias(alias: AliasDecl) -> Self {
        Self::Prim(PrimType::A(alias))
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

    pub fn tag(&self) -> Option<TypeTag> {
        match self {
            Type::Prim(p) => p.tag(),
            Type::Array(t, _) => t.tag(),
            Type::Tuple(ts) => containing_tag(ts.iter().map(Type::tag)),
            Type::Sum(ts) => containing_tag(ts.iter().map(Type::tag)),
        }
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
