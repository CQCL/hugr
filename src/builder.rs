//! Tools for building valid HUGRs.
//!
use thiserror::Error;

use crate::hugr::{HugrError, HugrMut, HugrView, Node, ValidationError, Wire};
use crate::ops::handle::{BasicBlockID, CfgID, ConditionalID, DfgID, FuncID, TailLoopID};
use crate::ops::DataflowOp;
use crate::types::{LinearType, Signature, TypeRow};
use crate::Hugr;

pub mod handle;
pub use handle::BuildHandle;

mod build_traits;
pub use build_traits::{Container, Dataflow};

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
    NoCopyLinear(LinearType),

    /// Error in CircuitBuilder
    #[error("Error in CircuitBuilder: {0}.")]
    CircuitError(#[from] circuit_builder::CircuitBuildError),
}

#[derive(Default)]
/// Base builder, can generate builders for containers
pub struct HugrBuilder {
    base: HugrMut,
}

impl HugrBuilder {
    /// Initialize a new builder
    pub fn new() -> Self {
        // initially assume to be a module root, will be replaced if not.
        Self {
            base: HugrMut::new_module(),
        }
    }

    /// Use this builder to build a module HUGR
    pub fn module_hugr_builder(&mut self) -> ModuleBuilder {
        ModuleBuilder(&mut self.base)
    }

    /// Use this builder to build a DFG HUGR
    pub fn dfg_hugr_builder(
        &mut self,
        input: impl Into<TypeRow>,
        output: impl Into<TypeRow>,
    ) -> Result<DFGBuilder, BuildError> {
        let input = input.into();
        let output = output.into();
        let root = self.base.hugr().root();
        let dfg_op = DataflowOp::DFG {
            signature: Signature::new_df(input.clone(), output.clone()),
        };
        self.base.replace_op(root, dfg_op);

        DFGBuilder::create_with_io(&mut self.base, root, input, output)
    }

    // TODO: CFG, BasicBlock, Def, Conditional, TailLoop, Case

    /// Complete building and return HUGR, performing validation.
    pub fn finish(self) -> Result<Hugr, BuildError> {
        Ok(self.base.finish()?)
    }
}

// macro to implement empty drop for builders
// forces use of value (through calling finish) before automatic drop
macro_rules! impl_builder_drop {
    ($name:ident) => {
        impl<'f> Drop for $name<'f> {
            fn drop(&mut self) {}
        }
    };
}
impl_builder_drop!(ModuleBuilder);
impl_builder_drop!(DFGBuilder);
impl_builder_drop!(CFGBuilder);
impl_builder_drop!(ConditionalBuilder);
impl<'f, T> Drop for DFGWrapper<'f, T> {
    fn drop(&mut self) {}
}

#[cfg(test)]
mod test {

    use crate::types::{ClassicType, LinearType, Signature, SimpleType};
    use crate::Hugr;

    use super::handle::BuildHandle;
    use super::{BuildError, Dataflow, FuncID, FunctionBuilder};
    use super::{Container, HugrBuilder};

    pub(super) const NAT: SimpleType = SimpleType::Classic(ClassicType::i64());
    pub(super) const F64: SimpleType = SimpleType::Classic(ClassicType::F64);
    pub(super) const BIT: SimpleType = SimpleType::Classic(ClassicType::bit());
    pub(super) const QB: SimpleType = SimpleType::Linear(LinearType::Qubit);

    /// Wire up inputs of a Dataflow container to the outputs.
    pub(super) fn n_identity<T: Dataflow>(
        dataflow_builder: T,
    ) -> Result<T::ContainerHandle, BuildError> {
        let w = dataflow_builder.input_wires();
        dataflow_builder.finish_with_outputs(w)
    }

    pub(super) fn build_main(
        signature: Signature,
        f: impl FnOnce(FunctionBuilder<true>) -> Result<BuildHandle<FuncID<true>>, BuildError>,
    ) -> Result<Hugr, BuildError> {
        let mut builder = HugrBuilder::new();
        let mut module_builder = builder.module_hugr_builder();
        let f_builder = module_builder.declare_and_def("main", signature)?;

        f(f_builder)?;
        module_builder.finish()?;
        builder.finish()
    }
}
