//! General wire types used in the compiler

pub mod custom;
mod signature;
pub mod simple;
pub mod type_param;
pub mod type_row;

pub use custom::CustomType;
pub use signature::{AbstractSignature, Signature, SignatureDescription};
pub use simple::{
    ClassicRow, ClassicType, Container, HashableType, PrimType, SimpleRow, SimpleType, TypeTag,
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
