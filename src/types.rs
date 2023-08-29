//! General wire types used in the compiler

mod check;
pub mod custom;
mod primitive;
mod serialize;
mod signature;
pub mod type_param;
pub mod type_row;

pub use check::{ConstTypeError, CustomCheckFailure};
pub use custom::CustomType;
pub use signature::{FunctionType, Signature, SignatureDescription};
pub use type_row::TypeRow;

use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::ops::AliasDecl;
use crate::type_row;
use std::fmt::Debug;

use self::primitive::PrimType;

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
        matches!(self, EdgeKind::Value(t) if !t.copyable())
    }
}

#[derive(
    Copy, Default, Clone, PartialEq, Eq, Hash, Debug, derive_more::Display, Serialize, Deserialize,
)]
/// Bounds on the valid operations on a type in a HUGR program.
pub enum TypeBound {
    /// The equality operation is valid on this type.
    #[serde(rename = "E")]
    Eq,
    /// The type can be copied in the program.
    #[serde(rename = "C")]
    Copyable,
    /// No bound on the type.
    #[serde(rename = "A")]
    #[default]
    Any,
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
    pub const fn contains(&self, other: TypeBound) -> bool {
        use TypeBound::*;
        matches!((self, other), (Any, _) | (_, Eq) | (Copyable, Copyable))
    }
}

/// Calculate the least upper bound for an iterator of bounds
pub(crate) fn least_upper_bound(mut tags: impl Iterator<Item = TypeBound>) -> TypeBound {
    tags.fold_while(TypeBound::Eq, |acc, new| {
        if acc == TypeBound::Any || new == TypeBound::Any {
            Done(TypeBound::Any)
        } else {
            Continue(acc.union(new))
        }
    })
    .into_inner()
}

#[derive(Clone, PartialEq, Debug, Eq, derive_more::Display, Serialize, Deserialize)]
#[serde(tag = "s")]
/// Representation of a Sum type.
/// Either store the types of the variants, or in the special (but common) case
/// of a "simple predicate" (sum over empty tuples), store only the size of the predicate.
enum SumType {
    #[display(fmt = "SimplePredicate({})", "size")]
    Simple {
        size: u8,
    },
    General {
        row: TypeRow,
    },
}

impl SumType {
    fn new(types: impl Into<TypeRow>) -> Self {
        let row: TypeRow = types.into();

        let len = row.len();
        if len <= (u8::MAX as usize) && row.iter().all(|t| *t == Type::UNIT) {
            Self::Simple { size: len as u8 }
        } else {
            Self::General { row: row }
        }
    }

    fn get_variant(&self, tag: usize) -> Option<&Type> {
        match self {
            SumType::Simple { size } if tag < (*size as usize) => Some(Type::UNIT_REF),
            SumType::General { row } => row.get(tag),
            _ => None,
        }
    }
}

impl From<SumType> for Type {
    fn from(sum: SumType) -> Type {
        match sum {
            SumType::Simple { size } => Type::new_simple_predicate(size),
            SumType::General { row: types } => Type::new_sum(types),
        }
    }
}

#[derive(Clone, PartialEq, Debug, Eq, derive_more::Display)]
/// Core types: primitive (leaf), tuple (product) or sum (co-product).
enum TypeEnum {
    Prim(PrimType),
    #[display(fmt = "Tuple({})", "_0")]
    Tuple(TypeRow),
    #[display(fmt = "Sum({})", "_0")]
    Sum(SumType),
}
impl TypeEnum {
    /// The smallest type bound that covers the whole type.
    fn least_upper_bound(&self) -> TypeBound {
        match self {
            TypeEnum::Prim(p) => p.bound(),
            TypeEnum::Sum(SumType::Simple { size: _ }) => TypeBound::Eq,
            TypeEnum::Sum(SumType::General { row: ts }) => {
                least_upper_bound(ts.iter().map(Type::least_upper_bound))
            }
            TypeEnum::Tuple(ts) => least_upper_bound(ts.iter().map(Type::least_upper_bound)),
        }
    }
}

#[derive(
    Clone, PartialEq, Debug, Eq, derive_more::Display, serde::Serialize, serde::Deserialize,
)]
#[display(fmt = "{}", "_0")]
#[serde(into = "serialize::SerSimpleType", from = "serialize::SerSimpleType")]
/// A HUGR type - the valid types of [EdgeKind::Value] and [EdgeKind::Static] edges.
/// Such an edge is valid if the ports on either end agree on the [Type].
/// Types have an optional [TypeBound] which places limits on the valid
/// operations on a type.
///
/// Examples:
/// ```
/// # use hugr::types::{Type, TypeBound};
/// # use hugr::type_row;
///
/// let sum = Type::new_sum(type_row![Type::UNIT, Type::UNIT]);
/// assert_eq!(sum.least_upper_bound(), TypeBound::Eq);
///
/// ```
///
/// ```
/// # use hugr::types::{Type, TypeBound, FunctionType};
///
/// let func_type = Type::new_function(FunctionType::new_linear(vec![]));
/// assert_eq!(func_type.least_upper_bound(), TypeBound::Copyable);
///
/// ```
///
pub struct Type(TypeEnum, TypeBound);

impl Type {
    /// Unit type (empty tuple).
    pub const UNIT: Self = Self(TypeEnum::Tuple(type_row![]), TypeBound::Eq);
    const UNIT_REF: &'static Self = &Self::UNIT;

    /// Initialize a new function type.
    pub fn new_function(signature: FunctionType) -> Self {
        Self::new(TypeEnum::Prim(PrimType::Function(Box::new(signature))))
    }

    /// Initialize a new tuple type by providing the elements.
    #[inline(always)]
    pub fn new_tuple(types: impl Into<TypeRow>) -> Self {
        Self::new(TypeEnum::Tuple(types.into()))
    }

    /// Initialize a new sum type by providing the possible variant types.
    #[inline(always)]
    pub fn new_sum(types: impl Into<TypeRow>) -> Self {
        Self::new(TypeEnum::Sum(SumType::new(types)))
    }

    /// Initialize a new custom type.
    // TODO remove? Extensions/TypeDefs should just provide `Type` directly
    pub const fn new_extension(opaque: CustomType) -> Self {
        let bound = opaque.bound();
        Type(TypeEnum::Prim(PrimType::Extension(opaque)), bound)
    }

    /// Initialize a new alias.
    pub fn new_alias(alias: AliasDecl) -> Self {
        Self::new(TypeEnum::Prim(PrimType::Alias(alias)))
    }

    fn new(type_e: TypeEnum) -> Self {
        let bound = type_e.least_upper_bound();
        Self(type_e, bound)
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
    pub const fn new_simple_predicate(size: u8) -> Self {
        // should be the only way to avoid going through SumType::new
        Self(TypeEnum::Sum(SumType::Simple { size: size }), TypeBound::Eq)
    }

    /// Report the least upper TypeBound, if there is one.
    #[inline(always)]
    pub const fn least_upper_bound(&self) -> TypeBound {
        self.1
    }

    /// Report if the type is copyable - i.e.the least upper bound of the type
    /// is contained by the copyable bound.
    pub const fn copyable(&self) -> bool {
        TypeBound::Copyable.contains(self.least_upper_bound())
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

#[cfg(test)]
pub(crate) mod test {

    use super::{
        custom::test::{ANY_CUST, COPYABLE_CUST, EQ_CUST},
        *,
    };
    use crate::{extension::prelude::USIZE_T, ops::AliasDecl};

    pub(crate) const EQ_T: Type = Type::new_extension(EQ_CUST);
    pub(crate) const COPYABLE_T: Type = Type::new_extension(COPYABLE_CUST);
    pub(crate) const ANY_T: Type = Type::new_extension(ANY_CUST);

    #[test]
    fn construct() {
        let t: Type = Type::new_tuple(vec![
            USIZE_T,
            Type::new_function(FunctionType::new_linear(vec![])),
            Type::new_extension(CustomType::new(
                "my_custom",
                [],
                "my_extension",
                TypeBound::Copyable,
            )),
            Type::new_alias(AliasDecl::new("my_alias", TypeBound::Eq)),
        ]);
        assert_eq!(
            t.to_string(),
            "Tuple([usize([]), Function([[]][]), my_custom([]), Alias(my_alias)])".to_string()
        );
    }

    #[test]
    fn sum_construct() {
        let pred1 = Type::new_sum(type_row![Type::UNIT, Type::UNIT]);
        let pred2 = Type::new_simple_predicate(2);

        assert_eq!(pred1, pred2);

        let pred_direct = SumType::Simple { size: 2 };
        assert_eq!(pred1, pred_direct.into())
    }
}
