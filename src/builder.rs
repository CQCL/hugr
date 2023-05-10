#![warn(missing_docs)]
//! Tools for building valid HUGRs.
//!
use portgraph::NodeIndex;
use thiserror::Error;

use crate::hugr::{HugrError, ValidationError};
use crate::types::LinearType;

pub mod nodehandle;
pub use nodehandle::{BasicBlockID, BuildHandle, CfgID, ConditionalID, DfgID, FuncID, TailLoopID};

mod build_traits;
pub use build_traits::{Container, Dataflow};

mod dataflow;
pub use dataflow::{DFGBuilder, FunctionBuilder};

mod module_builder;
pub use module_builder::ModuleBuilder;

mod cfg;
pub use cfg::{BlockBuilder, CFGBuilder};

mod tail_loop;
pub use tail_loop::TailLoopBuilder;

mod conditional;
pub use conditional::{CaseBuilder, ConditionalBuilder};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// A DataFlow wire, defined by a Value-kind output port of a node
// Stores node and offset to output port
pub struct Wire(NodeIndex, usize);

#[derive(Debug, Clone, PartialEq, Eq, Error)]
/// Error during building of HUGR
pub enum BuildError {
    /// The constructed HUGR is invalid.
    #[error("The constructed HUGR is invalid: {0}.")]
    InvalidHUGR(#[from] ValidationError),
    /// HUGR construction error.
    #[error("Error when mutating HUGR: {0}.")]
    ConstructError(#[from] HugrError),
    /// CFG can only have one entry.
    #[error("CFG entry node already built for CFG node: {0:?}.")]
    EntryBuiltError(NodeIndex),
    /// Node was expected to have a certain type but was found to not.
    #[error("Node with index {node:?} does not have type {op_desc:?} as expected.")]
    UnexpectedType {
        /// Index of node where error occurred
        node: NodeIndex,
        /// Description of expected node
        op_desc: &'static str,
    },
    /// Error building Conditional node
    #[error("Error building Conditional node: {0}.")]
    ConditionalError(#[from] conditional::ConditionalBuildError),

    /// Wire not found in Hugr
    #[error("Wire not found in Hugr: {0:?}.")]
    WireNotFound(Wire),

    /// Can't copy a linear type
    #[error("Can't copy linear type: {0:?}.")]
    NoCopyLinear(LinearType),
}

#[cfg(test)]
mod test {

    use crate::types::{ClassicType, LinearType, SimpleType};

    use super::{BuildError, Dataflow};

    pub(super) const NAT: SimpleType = SimpleType::Classic(ClassicType::i64());
    pub(super) const BIT: SimpleType = SimpleType::Classic(ClassicType::bit());
    pub(super) const QB: SimpleType = SimpleType::Linear(LinearType::Qubit);

    /// Wire up inputs of a Dataflow container to the outputs
    pub(super) fn n_identity<T: Dataflow>(
        dataflow_builder: T,
    ) -> Result<T::ContainerHandle, BuildError> {
        let w = dataflow_builder.input_wires();
        dataflow_builder.finish_with_outputs(w)
    }
}
