//! Tools for building valid HUGRs.
//!
use thiserror::Error;

use crate::hugr::{HugrError, Node, ValidationError, Wire};
use crate::ops::handle::{BasicBlockID, CfgID, ConditionalID, DfgID, FuncID, TailLoopID};
use crate::types::SimpleType;

pub mod handle;
pub use handle::BuildHandle;

mod build_traits;
pub use build_traits::{
    Container, Dataflow, DataflowHugr, DataflowSubContainer, HugrBuilder, SubContainer,
};

mod dataflow;
pub use dataflow::{DFGBuilder, DFGWrapper, FunctionBuilder};

mod module_builder;
pub use module_builder::ModuleBuilder;

mod cfg;
pub use cfg::{BlockBuilder, CFGBuilder};

mod tail_loop;
pub use tail_loop::TailLoopBuilder;

mod conditional;
pub use conditional::{CaseBuilder, ConditionalBuilder};

mod circuit_builder;
pub use circuit_builder::{AppendWire, CircuitBuilder};

#[derive(Debug, Clone, PartialEq, Eq, Error)]
/// Error while building the HUGR.
pub enum BuildError {
    /// The constructed HUGR is invalid.
    #[error("The constructed HUGR is invalid: {0}.")]
    InvalidHUGR(#[from] ValidationError),
    /// HUGR construction error.
    #[error("Error when mutating HUGR: {0}.")]
    ConstructError(#[from] HugrError),
    /// CFG can only have one entry.
    #[error("CFG entry node already built for CFG node: {0:?}.")]
    EntryBuiltError(Node),
    /// Node was expected to have a certain type but was found to not.
    #[error("Node with index {node:?} does not have type {op_desc:?} as expected.")]
    UnexpectedType {
        /// Index of node where error occurred.
        node: Node,
        /// Description of expected node.
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
    NoCopyLinear(SimpleType),

    /// Error in CircuitBuilder
    #[error("Error in CircuitBuilder: {0}.")]
    CircuitError(#[from] circuit_builder::CircuitBuildError),
}

#[cfg(test)]
mod test {
    use crate::types::{ClassicType, Signature, SimpleType};
    use crate::Hugr;

    use super::handle::BuildHandle;
    use super::{BuildError, Container, FuncID, FunctionBuilder, ModuleBuilder};
    use super::{DataflowSubContainer, HugrBuilder};

    pub(super) const NAT: SimpleType = SimpleType::Classic(ClassicType::i64());
    pub(super) const F64: SimpleType = SimpleType::Classic(ClassicType::F64);
    pub(super) const BIT: SimpleType = SimpleType::Classic(ClassicType::bit());
    pub(super) const QB: SimpleType = SimpleType::Qubit;

    /// Wire up inputs of a Dataflow container to the outputs.
    pub(super) fn n_identity<T: DataflowSubContainer>(
        dataflow_builder: T,
    ) -> Result<T::ContainerHandle, BuildError> {
        let w = dataflow_builder.input_wires();
        dataflow_builder.finish_with_outputs(w)
    }

    pub(super) fn build_main(
        signature: Signature,
        f: impl FnOnce(FunctionBuilder<&mut Hugr>) -> Result<BuildHandle<FuncID<true>>, BuildError>,
    ) -> Result<Hugr, BuildError> {
        let mut module_builder = ModuleBuilder::new();
        let f_builder = module_builder.define_function("main", signature)?;

        f(f_builder)?;
        Ok(module_builder.finish_hugr()?)
    }
}
