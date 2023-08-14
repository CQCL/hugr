//! General wire types used in the compiler

mod check;
pub mod custom;
mod primitive;
mod serialize;
mod signature;
pub mod type_param;
pub mod type_row;

pub use check::{ConstTypeError, CustomCheckFail};
pub use custom::CustomType;
pub use signature::{AbstractSignature, Signature, SignatureDescription};
pub use type_row::TypeRow;

use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{ops::AliasDecl, resource::PRELUDE};
use std::fmt::Debug;

use self::primitive::PrimType;
use self::type_param::TypeArg;

/// The kinds of edges in a HUGR, excluding Hierarchy.
//#[cfg_attr(feature = "pyo3", pyclass)] # TODO: Manually derive pyclass with non-unit variants
#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum EdgeKind {
    /// Control edges of a CFG region.
    ControlFlow,
    /// Data edges of a DDG region, also known as "wires".
    Value(Type),
    /// A reference to a static value definition.
    Static(Type),
    /// Explicitly enforce an ordering between nodes in a DDG.
    StateOrder,
}

impl EdgeKind {
    /// Returns whether the type might contain linear data.
    pub fn is_linear(&self) -> bool {
        match self {
            EdgeKind::Value(t) => t.least_upper_bound().is_some(),
            _ => false,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, derive_more::Display, Serialize, Deserialize)]
/// Bounds on capabilities of a type.
pub enum TypeBound {
    /// The equality operation is valid on this type.
    #[serde(rename = "E")]
    Eq,
    /// The type can be copied in the program.
    #[serde(rename = "C")]
    Copyable,
}

impl TypeBound {
    /// Returns the smallest TypeTag containing both the receiver and argument.
    /// (This will be one of the receiver or the argument.)
    pub fn union(self, other: Self) -> Self {
        if self.contains(other) {
            self
        } else {
            debug_assert!(other.contains(self));
            other
        }
    }

    /// Report if this bound contains another.
    pub fn contains(&self, other: TypeBound) -> bool {
        use TypeBound::*;
        match (self, other) {
            (Copyable, Eq) => true,
            (Eq, Copyable) => false,
            _ => true,
        }
    }
}

/// Calculate the least upper bound for an iterator of bounds
pub(crate) fn least_upper_bound(
    mut tags: impl Iterator<Item = Option<TypeBound>>,
) -> Option<TypeBound> {
    tags.fold_while(Some(TypeBound::Eq), |acc, new| {
        if let (Some(acc), Some(new)) = (acc, new) {
            Continue(Some(acc.union(new)))
        } else {
            // if any type is unbounded, short-circuit
            Done(None)
        }
    })
    .into_inner()
}

#[derive(Clone, PartialEq, Debug, Eq, derive_more::Display)]
enum TypeEnum {
    Prim(PrimType),
    #[display(fmt = "Tuple({})", "_0")]
    Tuple(TypeRow),
    #[display(fmt = "Sum({})", "_0")]
    Sum(TypeRow),
}
impl TypeEnum {
    fn least_upper_bound(&self) -> Option<TypeBound> {
        match self {
            TypeEnum::Prim(p) => p.bound(),
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
/// A HUGR type.
pub struct Type(TypeEnum, Option<TypeBound>);

impl Type {
    /// Initialize a new graph type with a signature.
    pub fn graph(signature: AbstractSignature) -> Self {
        Self::new(TypeEnum::Prim(PrimType::Graph(Box::new(signature))))
    }

    /// Initialize a new usize type.
    pub fn usize() -> Self {
        Self::new_extension(
            PRELUDE
                .get_type("usize")
                .unwrap()
                .instantiate_concrete(vec![])
                .unwrap(),
        )
    }

    /// Initialize a new tuple type by providing the elements..
    pub fn new_tuple(types: impl Into<TypeRow>) -> Self {
        Self::new(TypeEnum::Tuple(types.into()))
    }

    /// New unit type, defined as an empty Tuple.
    pub fn new_unit() -> Self {
        Self::new_tuple(vec![])
    }

    /// Initialize a new sum type by providing the possible variant types.
    pub fn new_sum(types: impl Into<TypeRow>) -> Self {
        Self::new(TypeEnum::Sum(types.into()))
    }

    /// Initialize a new custom type.
    // TODO remove? Resources/TypeDefs should just provide `Type` directly
    pub fn new_extension(opaque: CustomType) -> Self {
        Self::new(TypeEnum::Prim(PrimType::E(opaque)))
    }

    /// Initialize a new alias.
    pub fn new_alias(alias: AliasDecl) -> Self {
        Self::new(TypeEnum::Prim(PrimType::A(alias)))
    }

    fn new(type_e: TypeEnum) -> Self {
        let bound = type_e.least_upper_bound();
        Self(type_e, bound)
    }

    /// Initialize a new array of type `typ` of length `size`
    pub fn new_array(typ: Type, size: u64) -> Self {
        let array_def = PRELUDE.get_type("array").unwrap();
        let custom_t = array_def
            .instantiate_concrete(vec![TypeArg::Type(typ), TypeArg::USize(size)])
            .unwrap();
        Self::new_extension(custom_t)
    }
    /// New Sum of Tuple types, used as predicates in branching.
    /// Tuple rows are defined in order by input rows.
    pub fn new_predicate<V>(variant_rows: impl IntoIterator<Item = V>) -> Self
    where
        V: Into<TypeRow>,
    {
        Self::new_sum(predicate_variants_row(variant_rows))
    }

    /// New simple predicate with empty Tuple variants
    pub fn new_simple_predicate(size: usize) -> Self {
        Self::new_predicate(std::iter::repeat(vec![]).take(size))
    }

    /// Report the least upper TypeBound, if there is one.
    pub fn least_upper_bound(&self) -> Option<TypeBound> {
        self.1
    }
}

/// Return the type row of variants required to define a Sum of Tuples type
/// given the rows of each tuple
pub(crate) fn predicate_variants_row<V>(variant_rows: impl IntoIterator<Item = V>) -> TypeRow
where
    V: Into<TypeRow>,
{
    variant_rows
        .into_iter()
        .map(Type::new_tuple)
        .collect_vec()
        .into()
}

pub(crate) const ERROR_TYPE: Type = Type(
    TypeEnum::Prim(primitive::PrimType::E(CustomType::new_simple(
        smol_str::SmolStr::new_inline("error"),
        smol_str::SmolStr::new_inline("MyRsrc"),
        Some(TypeBound::Eq),
    ))),
    Some(TypeBound::Copyable),
);

#[cfg(test)]
pub(crate) mod test {

    use super::{
        custom::test::{ANY_CUST, COPYABLE_CUST, EQ_CUST},
        primitive::PrimType,
        *,
    };
    use crate::ops::AliasDecl;

    pub(crate) const EQ_T: Type = Type(
        TypeEnum::Prim(PrimType::E(EQ_CUST)),
        Some(TypeBound::Copyable),
    );

    pub(crate) const COPYABLE_T: Type = Type(
        TypeEnum::Prim(PrimType::E(COPYABLE_CUST)),
        Some(TypeBound::Copyable),
    );

    pub(crate) const ANY_T: Type = Type(
        TypeEnum::Prim(PrimType::E(ANY_CUST)),
        Some(TypeBound::Copyable),
    );

    #[test]
    fn construct() {
        let t: Type = Type::new_tuple(vec![
            Type::usize(),
            Type::graph(AbstractSignature::new_linear(vec![])),
            Type::new_extension(CustomType::new(
                "my_custom",
                [],
                "my_resource",
                Some(TypeBound::Copyable),
            )),
            Type::new_alias(AliasDecl::new("my_alias", Some(TypeBound::Eq))),
        ]);
        assert_eq!(
            t.to_string(),
            "Tuple([usize([]), Graph([[]][]), my_custom([]), Alias(my_alias)])".to_string()
        );
    }
}
