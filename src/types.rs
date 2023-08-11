//! General wire types used in the compiler

pub mod custom;
mod primitive;
mod serialize;
mod signature;
pub mod simple;
pub mod type_param;
pub mod type_row;

pub use custom::CustomType;
use serde_repr::{Deserialize_repr, Serialize_repr};
pub use signature::{AbstractSignature, Signature, SignatureDescription};
pub use simple::{
    ClassicRow, ClassicType, Container, HashableType, PrimType, SimpleRow, SimpleType,
};
pub use type_row::TypeRow;

/// The kinds of edges in a HUGR, excluding Hierarchy.
//#[cfg_attr(feature = "pyo3", pyclass)] # TODO: Manually derive pyclass with non-unit variants
#[derive(Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum EdgeKind {
    /// Control edges of a CFG region.
    ControlFlow,
    /// Data edges of a DDG region, also known as "wires".
    Value(SimpleType),
    /// A reference to a static value definition.
    Static(ClassicType),
    /// Explicitly enforce an ordering between nodes in a DDG.
    StateOrder,
}

impl EdgeKind {
    /// Returns whether the type might contain linear data.
    pub fn is_linear(&self) -> bool {
        match self {
            EdgeKind::Value(t) => !t.tag().is_classical(),
            _ => false,
        }
    }
}
/// Categorizes types into three classes according to basic operations supported.
#[derive(
    Copy, Clone, PartialEq, Eq, Hash, Debug, derive_more::Display, Serialize_repr, Deserialize_repr,
)]
#[repr(u8)]
pub enum TypeTag {
    /// Any [SimpleType], including linear and quantum types;
    /// cannot necessarily be copied or discarded.
    Simple = 0,
    /// Subset of [TypeTag::Simple]; types that can be copied and discarded. See [ClassicType]
    Classic = 1,
    /// Subset of [TypeTag::Classic]: types that can also be hashed and support
    /// a strong notion of equality. See [HashableType]
    Hashable = 2,
}

impl TypeTag {
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

    /// Do types in this tag contain only classic data
    /// (which can be copied and discarded, i.e. [ClassicType]s)
    pub fn is_classical(self) -> bool {
        self != Self::Simple
    }

    /// Do types in this tag contain only hashable classic data
    /// (with a strong notion of equality, i.e. [HashableType]s)
    pub fn is_hashable(self) -> bool {
        self == Self::Hashable
    }

    /// Report if this tag contains another.
    pub fn contains(&self, other: TypeTag) -> bool {
        use TypeTag::*;
        matches!(
            (self, other),
            (Simple, _) | (_, Hashable) | (Classic, Classic)
        )
    }
}

use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use serde::{Deserialize, Serialize};

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
fn least_upper_bound(mut tags: impl Iterator<Item = Option<TypeBound>>) -> Option<TypeBound> {
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

use std::fmt::Write;

use crate::{ops::AliasDecl, resource::PRELUDE, utils::display_list};
use std::fmt::{self, Debug, Display};

use self::type_param::TypeArg;

//TODO remove
type NewPrimType = primitive::PrimType;

#[derive(Clone, PartialEq, Debug, Eq, derive_more::Display)]
enum TypeEnum {
    Prim(NewPrimType),
    #[display(fmt = "Tuple({})", "DisplayRow(_0)")]
    Tuple(Vec<Type>),
    #[display(fmt = "Sum({})", "DisplayRow(_0)")]
    Sum(Vec<Type>),
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

struct DisplayRow<'a>(&'a Vec<Type>);
impl<'a> Display for DisplayRow<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;
        display_list(self.0, f)?;
        f.write_char(']')
    }
}

impl Type {
    /// Initialize a new graph type with a signature.
    pub fn graph(signature: AbstractSignature) -> Self {
        Self::new(TypeEnum::Prim(NewPrimType::Graph(Box::new(signature))))
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
    pub fn new_tuple(types: impl IntoIterator<Item = Type>) -> Self {
        Self::new(TypeEnum::Tuple(types.into_iter().collect()))
    }

    /// New unit type, defined as an empty Tuple.
    pub fn new_unit() -> Self {
        Self::new_tuple(vec![])
    }

    /// Initialize a new sum type by providing the possible variant types.
    pub fn new_sum(types: impl IntoIterator<Item = Type>) -> Self {
        Self::new(TypeEnum::Sum(types.into_iter().collect()))
    }

    /// Initialize a new custom type.
    // TODO remove? Resources/TypeDefs should just provide `Type` directly
    pub fn new_extension(opaque: CustomType) -> Self {
        Self::new(TypeEnum::Prim(NewPrimType::E(Box::new(opaque))))
    }

    /// Initialize a new alias.
    pub fn new_alias(alias: AliasDecl) -> Self {
        Self::new(TypeEnum::Prim(NewPrimType::A(alias)))
    }

    fn new(type_e: TypeEnum) -> Self {
        let bound = type_e.least_upper_bound();
        Self(type_e, bound)
    }

    /// Initialize a new array of type `typ` of length `size`
    pub fn new_array(typ: Type, size: u64) -> Self {
        let array_def = PRELUDE.get_type("array").unwrap();
        // TODO replace with new Type
        let custom_t = array_def
            .instantiate_concrete(vec![TypeArg::Type(todo!()), TypeArg::USize(size)])
            .unwrap();
        Self::new_extension(custom_t)
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

    /// Report the least upper TypeBound, if there is one.
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

#[cfg(test)]
mod test {
    use crate::ops::AliasDecl;

    use super::*;
    #[test]
    fn construct() {
        let t: Type = Type::new_tuple([
            Type::usize(),
            Type::graph(AbstractSignature::new_linear(vec![])),
            Type::new_extension(CustomType::new(
                "my_custom",
                [],
                "my_resource",
                TypeTag::Classic,
            )),
            Type::new_alias(AliasDecl::new("my_alias", TypeTag::Hashable)),
        ]);
        assert_eq!(
            t.to_string(),
            "Tuple([usize([]), Graph([[]][]), my_custom([]), Alias(my_alias)])".to_string()
        );
    }
}
