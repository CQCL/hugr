//! Tools for building valid HUGRs.
//!
use thiserror::Error;

#[cfg(feature = "pyo3")]
use pyo3::{create_exception, exceptions::PyException, PyErr};

use crate::hugr::{HugrError, Node, ValidationError, Wire};
use crate::ops::handle::{BasicBlockID, CfgID, ConditionalID, DfgID, FuncID, TailLoopID};
use crate::types::ConstTypeError;
use crate::types::Type;

pub mod handle;
pub use handle::BuildHandle;

mod build_traits;
pub use build_traits::{
    Container, Dataflow, DataflowHugr, DataflowSubContainer, HugrBuilder, SubContainer,
};

mod dataflow;
pub use dataflow::{DFGBuilder, DFGWrapper, FunctionBuilder};

mod module;
pub use module::ModuleBuilder;

mod cfg;
pub use cfg::{BlockBuilder, CFGBuilder};

mod tail_loop;
pub use tail_loop::TailLoopBuilder;

mod conditional;
pub use conditional::{CaseBuilder, ConditionalBuilder};

mod circuit;
pub use circuit::CircuitBuilder;

#[derive(Debug, Clone, PartialEq, Error)]
/// Error while building the HUGR.
pub enum BuildError {
    /// The constructed HUGR is invalid.
    #[error("The constructed HUGR is invalid: {0}.")]
    InvalidHUGR(#[from] ValidationError),
    /// Tried to add a malformed [Const]
    ///
    /// [Const]: crate::ops::constant::Const
    #[error("Constant failed typechecking: {0}")]
    BadConstant(#[from] ConstTypeError),
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
    NoCopyLinear(Type),

    /// Error in CircuitBuilder
    #[error("Error in CircuitBuilder: {0}.")]
    CircuitError(#[from] circuit::CircuitBuildError),
}

#[cfg(feature = "pyo3")]
create_exception!(
    pyrs,
    PyBuildError,
    PyException,
    "Errors that can occur while building a Hugr"
);

#[cfg(feature = "pyo3")]
impl From<BuildError> for PyErr {
    fn from(err: BuildError) -> Self {
        PyBuildError::new_err(err.to_string())
    }
}

#[cfg(test)]
pub(crate) mod test {
    use rstest::fixture;

    use crate::hugr::{views::HugrView, HugrMut, NodeType};
    use crate::ops;
    use crate::types::{FunctionType, Signature, Type};
    use crate::{type_row, Hugr};

    use super::handle::BuildHandle;
    use super::{
        BuildError, Container, DFGBuilder, Dataflow, DataflowHugr, FuncID, FunctionBuilder,
        ModuleBuilder,
    };
    use super::{DataflowSubContainer, HugrBuilder};

    pub(super) const NAT: Type = crate::extension::prelude::USIZE_T;
    pub(super) const BIT: Type = crate::extension::prelude::BOOL_T;
    pub(super) const QB: Type = crate::extension::prelude::QB_T;

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
        Ok(module_builder.finish_prelude_hugr()?)
    }

    #[fixture]
    pub(crate) fn simple_dfg_hugr() -> Hugr {
        let dfg_builder =
            DFGBuilder::new(FunctionType::new(type_row![BIT], type_row![BIT])).unwrap();
        let [i1] = dfg_builder.input_wires_arr();
        dfg_builder.finish_prelude_hugr_with_outputs([i1]).unwrap()
    }

    /// A helper method which creates a DFG rooted hugr with closed resources,
    /// for tests which want to avoid having open extension variables after
    /// inference. Using DFGBuilder will default to a root node with an open
    /// extension variable
    pub(crate) fn closed_dfg_root_hugr(signature: FunctionType) -> Hugr {
        let mut hugr = Hugr::new(NodeType::pure(ops::DFG {
            signature: signature.clone(),
        }));
        hugr.add_node_with_parent(
            hugr.root(),
            NodeType::open_extensions(ops::Input {
                types: signature.input,
            }),
        )
        .unwrap();
        hugr.add_node_with_parent(
            hugr.root(),
            NodeType::open_extensions(ops::Output {
                types: signature.output,
            }),
        )
        .unwrap();
        hugr
    }
}
