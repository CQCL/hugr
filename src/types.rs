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
pub use signature::{FunctionType, Signature};
pub use type_param::TypeArg;
pub use type_row::TypeRow;

use derive_more::{From, Into};
use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::extension::{ExtensionRegistry, SignatureError};
use crate::ops::AliasDecl;
use crate::type_row;
use std::fmt::Debug;

pub use self::primitive::PrimType;

#[cfg(feature = "pyo3")]
use pyo3::pyclass;

/// The kinds of edges in a HUGR, excluding Hierarchy.
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

/// Python representation for [`EdgeKind`], the kinds of edges in a HUGR.
#[cfg_attr(feature = "pyo3", pyclass)]
#[repr(transparent)]
#[derive(Clone, PartialEq, Eq, Debug, From, Into)]
pub struct PyEdgeKind(EdgeKind);

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
/// of a UnitSum (sum over empty tuples), store only the size of the Sum.
pub enum SumType {
    #[allow(missing_docs)]
    #[display(fmt = "UnitSum({})", "size")]
    Unit { size: u8 },
    #[allow(missing_docs)]
    General { row: TypeRow },
}

impl SumType {
    /// Initialize a new sum type.
    pub fn new(types: impl Into<TypeRow>) -> Self {
        let row: TypeRow = types.into();

        let len: usize = row.len();
        if len <= (u8::MAX as usize) && row.iter().all(|t| *t == Type::UNIT) {
            Self::Unit { size: len as u8 }
        } else {
            Self::General { row }
        }
    }

    /// Report the tag'th variant, if it exists.
    pub fn get_variant(&self, tag: usize) -> Option<&Type> {
        match self {
            SumType::Unit { size } if tag < (*size as usize) => Some(Type::UNIT_REF),
            SumType::General { row } => row.get(tag),
            _ => None,
        }
    }
}

impl From<SumType> for Type {
    fn from(sum: SumType) -> Type {
        match sum {
            SumType::Unit { size } => Type::new_unit_sum(size),
            SumType::General { row } => Type::new_sum(row),
        }
    }
}

#[derive(Clone, PartialEq, Debug, Eq, derive_more::Display)]
/// Core types: primitive (leaf), tuple (product) or sum (co-product).
pub enum TypeEnum {
    #[allow(missing_docs)]
    Prim(PrimType),
    #[allow(missing_docs)]
    #[display(fmt = "Tuple({})", "_0")]
    Tuple(TypeRow),
    #[allow(missing_docs)]
    #[display(fmt = "Sum({})", "_0")]
    Sum(SumType),
}
impl TypeEnum {
    /// The smallest type bound that covers the whole type.
    fn least_upper_bound(&self) -> TypeBound {
        match self {
            TypeEnum::Prim(p) => p.bound(),
            TypeEnum::Sum(SumType::Unit { size: _ }) => TypeBound::Eq,
            TypeEnum::Sum(SumType::General { row }) => {
                least_upper_bound(row.iter().map(Type::least_upper_bound))
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
#[cfg_attr(feature = "pyo3", pyclass)]
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

    /// New Sum of Tuple types, used in branching control.
    /// Tuple rows are defined in order by input rows.
    pub fn new_tuple_sum<V>(variant_rows: impl IntoIterator<Item = V>) -> Self
    where
        V: Into<TypeRow>,
    {
        Self::new_sum(tuple_sum_row(variant_rows))
    }

    /// New UnitSum with empty Tuple variants
    pub const fn new_unit_sum(size: u8) -> Self {
        // should be the only way to avoid going through SumType::new
        Self(TypeEnum::Sum(SumType::Unit { size }), TypeBound::Eq)
    }

    /// Report the least upper TypeBound, if there is one.
    #[inline(always)]
    pub const fn least_upper_bound(&self) -> TypeBound {
        self.1
    }

    /// Report the component TypeEnum.
    #[inline(always)]
    pub const fn as_type_enum(&self) -> &TypeEnum {
        &self.0
    }

    /// Report if the type is copyable - i.e.the least upper bound of the type
    /// is contained by the copyable bound.
    pub const fn copyable(&self) -> bool {
        TypeBound::Copyable.contains(self.least_upper_bound())
    }

    pub(crate) fn validate(
        &self,
        extension_registry: &ExtensionRegistry,
    ) -> Result<(), SignatureError> {
        // There is no need to check the components against the bound,
        // that is guaranteed by construction (even for deserialization)
        match &self.0 {
            TypeEnum::Tuple(row) | TypeEnum::Sum(SumType::General { row }) => {
                row.iter().try_for_each(|t| t.validate(extension_registry))
            }
            TypeEnum::Sum(SumType::Unit { .. }) => Ok(()), // No leaves there
            TypeEnum::Prim(PrimType::Alias(_)) => Ok(()),
            TypeEnum::Prim(PrimType::Extension(custy)) => custy.validate(extension_registry),
            TypeEnum::Prim(PrimType::Function(ft)) => ft
                .input
                .iter()
                .chain(ft.output.iter())
                .try_for_each(|t| t.validate(extension_registry)),
        }
    }
}

/// Return the type row of variants required to define a Sum of Tuples type
/// given the rows of each tuple
pub(crate) fn tuple_sum_row<V>(variant_rows: impl IntoIterator<Item = V>) -> TypeRow
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

    use super::*;
    use crate::{
        extension::{prelude::USIZE_T, PRELUDE},
        ops::AliasDecl,
        std_extensions::arithmetic::float_types,
    };

    use crate::types::TypeBound;

    pub(crate) fn test_registry() -> ExtensionRegistry {
        vec![PRELUDE.to_owned(), float_types::extension()].into()
    }

    #[test]
    fn construct() {
        let t: Type = Type::new_tuple(vec![
            USIZE_T,
            Type::new_function(FunctionType::new_linear(vec![])),
            Type::new_extension(CustomType::new(
                "my_custom",
                [],
                "my_extension".try_into().unwrap(),
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
        let pred2 = Type::new_unit_sum(2);

        assert_eq!(pred1, pred2);

        let pred_direct = SumType::Unit { size: 2 };
        assert_eq!(pred1, pred_direct.into())
    }
}
