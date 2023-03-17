//! Types used in the compiler

pub mod angle;
pub mod dataflow;
pub mod resource;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

pub use angle::{AngleValue, Quat, Rational};
pub use dataflow::{DataType, RowType};
pub use resource::{Resource, ResourceValue};

/// The wire types
//#[cfg_attr(feature = "pyo3", pyclass)] # TODO: Manually derive pyclass with non-unit variants
#[derive(Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum Type {
    /// Control edges of a CFG region
    ControlFlow,
    /// Data edges of a DDG region
    Dataflow(DataType),
    /// A reference to a constant value definition, used in the module region.
    Const(DataType),
    /// A strict ordering between nodes
    StateOrder,
}

impl Default for Type {
    fn default() -> Self {
        Self::ControlFlow
    }
}

/// A function signature
#[cfg_attr(feature = "pyo3", pyclass)]
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Signature {
    /// Input of the function
    pub input: RowType,
    /// Output of the function
    pub output: RowType,
    /// Constant data references used by the function
    pub consts: RowType,
}

#[cfg_attr(feature = "pyo3", pymethods)]
impl Signature {
    /// The number of wires in the signature
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.consts.is_empty() && self.input.is_empty() && self.output.is_empty()
    }

    /// Returns whether the data wires in the signature are purely linear
    #[inline(always)]
    pub fn purely_linear(&self) -> bool {
        self.input.purely_linear() && self.output.purely_linear()
    }

    /// Returns whether the data wires in the signature are purely classical
    #[inline(always)]
    pub fn purely_classical(&self) -> bool {
        self.input.purely_classical() && self.output.purely_classical()
    }
}

impl Signature {
    /// Create a new signature
    pub fn new(
        input: impl Into<RowType>,
        output: impl Into<RowType>,
        consts: impl Into<RowType>,
    ) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            consts: consts.into(),
        }
    }
}
