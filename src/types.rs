//! General wire types used in the compiler

pub mod custom;
mod primitive;
mod signature;
pub mod simple;
pub mod type_enum;
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

#[cfg(test)]
mod test {
    use crate::{ops::AliasDecl, types::type_enum::Type};

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
