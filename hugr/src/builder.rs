//! Utilities for building valid HUGRs.
//!
//! This module includes various tools for building HUGRs.
//!
//! Depending on the type of HUGR you want to build, you may want to use one of
//! the following builders:
//!
//! - [ModuleBuilder]: For building a module with function declarations and
//!       definitions.
//! - [DFGBuilder]: For building a dataflow graph.
//! - [FunctionBuilder]: A `DFGBuilder` specialised in defining functions with a
//!       dataflow graph.
//! - [CFGBuilder]: For building a control flow graph.
//! - [ConditionalBuilder]: For building a conditional node.
//! - [TailLoopBuilder]: For building a tail-loop node.
//!
//! Additionally, the [CircuitBuilder] provides an alternative to the
//! [DFGBuilder] when working with circuits, where some inputs of operations directly
//! correspond to some outputs and operations can be directly appended using
//! unit indices.
//!
//! # Example
//!
//! The following example shows how to build a simple HUGR module with two
//! dataflow functions, one built using the `DFGBuilder` and the other using the
//! `CircuitBuilder`.
//!
//! ```rust
//! # use hugr::Hugr;
//! # use hugr::builder::{BuildError, BuildHandle, Container, DFGBuilder, Dataflow, DataflowHugr, ModuleBuilder, DataflowSubContainer, HugrBuilder};
//! use hugr::extension::prelude::BOOL_T;
//! use hugr::std_extensions::logic::{NotOp, LOGIC_REG};
//! use hugr::types::FunctionType;
//!
//! # fn doctest() -> Result<(), BuildError> {
//! let hugr = {
//!     let mut module_builder = ModuleBuilder::new();
//!
//!     // Add a `main` function with signature `bool -> bool`.
//!     //
//!     // This block returns a handle to the built function.
//!     let _dfg_handle = {
//!         let mut dfg = module_builder.define_function(
//!             "main",
//!             FunctionType::new(vec![BOOL_T], vec![BOOL_T]).into(),
//!         )?;
//!
//!         // Get the wires from the function inputs.
//!         let [w] = dfg.input_wires_arr();
//!
//!         // Add an operation connected to the input wire, and get the new dangling wires.
//!         let [w] = dfg.add_dataflow_op(NotOp, [w])?.outputs_arr();
//!
//!         // Finish the function, connecting some wires to the output.
//!         dfg.finish_with_outputs([w])
//!     }?;
//!
//!     // Add a similar function, using the circuit builder interface.
//!     let _circuit_handle = {
//!         let mut dfg = module_builder.define_function(
//!             "circuit",
//!             FunctionType::new(vec![BOOL_T, BOOL_T], vec![BOOL_T, BOOL_T]).into(),
//!         )?;
//!         let mut circuit = dfg.as_circuit(dfg.input_wires());
//!
//!         // Add multiple operations, indicating only the wire index.
//!         circuit.append(NotOp, [0])?.append(NotOp, [1])?;
//!
//!         // Finish the circuit, and return the dataflow graph after connecting its outputs.
//!         let outputs = circuit.finish();
//!         dfg.finish_with_outputs(outputs)
//!     }?;
//!
//!     // Finish building the HUGR, consuming the builder.
//!     //
//!     // Requires a registry with all the extensions used in the module.
//!     module_builder.finish_hugr(&LOGIC_REG)
//! }?;
//!
//! // The built HUGR is always valid.
//! hugr.validate(&LOGIC_REG).unwrap_or_else(|e| {
//!     panic!("HUGR validation failed: {e}");
//! });
//! # Ok(())
//! # }
//! # doctest().unwrap();
//! ```
//!
use thiserror::Error;

use crate::extension::SignatureError;
use crate::hugr::ValidationError;
use crate::ops::handle::{BasicBlockID, CfgID, ConditionalID, DfgID, FuncID, TailLoopID};
use crate::ops::{NamedOp, OpType};
use crate::types::ConstTypeError;
use crate::types::Type;
use crate::{Node, Port, Wire};

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
pub use circuit::{CircuitBuildError, CircuitBuilder};

#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
/// Error while building the HUGR.
pub enum BuildError {
    /// The constructed HUGR is invalid.
    #[error("The constructed HUGR is invalid: {0}.")]
    InvalidHUGR(#[from] ValidationError),
    /// SignatureError in trying to construct a node (differs from
    /// [ValidationError::SignatureError] in that we could not construct a node to report about)
    #[error(transparent)]
    SignatureError(#[from] SignatureError),
    /// Tried to add a malformed [Const]
    ///
    /// [Const]: crate::ops::constant::Const
    #[error("Constant failed typechecking: {0}")]
    BadConstant(#[from] ConstTypeError),
    /// CFG can only have one entry.
    #[error("CFG entry node already built for CFG node: {0:?}.")]
    EntryBuiltError(Node),
    /// Node was expected to have a certain type but was found to not.
    #[error("Node with index {node:?} does not have type {op_desc:?} as expected.")]
    #[allow(missing_docs)]
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

    /// Error in CircuitBuilder
    #[error("Error in CircuitBuilder: {0}.")]
    CircuitError(#[from] circuit::CircuitBuildError),

    /// Invalid wires when setting outputs
    #[error("Found an error while setting the outputs of a {} container, {container_node}. {error}", .container_op.name())]
    #[allow(missing_docs)]
    OutputWiring {
        container_op: OpType,
        container_node: Node,
        #[source]
        error: BuilderWiringError,
    },

    /// Invalid input wires to a new operation
    ///
    /// The internal error message already contains the node index.
    #[error("Got an input wire while adding a {} to the circuit. {error}", .op.name())]
    #[allow(missing_docs)]
    OperationWiring {
        op: OpType,
        #[source]
        error: BuilderWiringError,
    },
}

#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
/// Error raised when wiring up a node during the build process.
pub enum BuilderWiringError {
    /// Tried to copy a linear type.
    #[error("Cannot copy linear type {typ} from output {src_offset} of node {src}")]
    #[allow(missing_docs)]
    NoCopyLinear {
        typ: Type,
        src: Node,
        src_offset: Port,
    },
    /// The ancestors of an inter-graph edge are not related.
    #[error("Cannot connect an inter-graph edge between unrelated nodes. Tried connecting {src} ({src_offset}) with {dst} ({dst_offset}).")]
    #[allow(missing_docs)]
    NoRelationIntergraph {
        src: Node,
        src_offset: Port,
        dst: Node,
        dst_offset: Port,
    },
    /// Inter-Graph edges can only carry copyable data.
    #[error("Inter-graph edges cannot carry non-copyable data {typ}. Tried connecting {src} ({src_offset}) with {dst} ({dst_offset}).")]
    #[allow(missing_docs)]
    NonCopyableIntergraph {
        src: Node,
        src_offset: Port,
        dst: Node,
        dst_offset: Port,
        typ: Type,
    },
}

#[cfg(test)]
pub(crate) mod test {
    use rstest::fixture;

    use crate::hugr::{views::HugrView, HugrMut, NodeType};
    use crate::ops;
    use crate::types::{FunctionType, PolyFuncType, Type};
    use crate::{type_row, Hugr};

    use super::handle::BuildHandle;
    use super::{
        BuildError, CFGBuilder, Container, DFGBuilder, Dataflow, DataflowHugr, FuncID,
        FunctionBuilder, ModuleBuilder,
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
        signature: PolyFuncType,
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

    #[fixture]
    pub(crate) fn simple_cfg_hugr() -> Hugr {
        let mut cfg_builder =
            CFGBuilder::new(FunctionType::new(type_row![NAT], type_row![NAT])).unwrap();
        super::cfg::test::build_basic_cfg(&mut cfg_builder).unwrap();
        cfg_builder.finish_prelude_hugr().unwrap()
    }

    /// A helper method which creates a DFG rooted hugr with closed resources,
    /// for tests which want to avoid having open extension variables after
    /// inference. Using DFGBuilder will default to a root node with an open
    /// extension variable
    pub(crate) fn closed_dfg_root_hugr(signature: FunctionType) -> Hugr {
        let mut hugr = Hugr::new(NodeType::new_pure(ops::DFG {
            signature: signature.clone(),
        }));
        hugr.add_node_with_parent(
            hugr.root(),
            ops::Input {
                types: signature.input,
            },
        );
        hugr.add_node_with_parent(
            hugr.root(),
            ops::Output {
                types: signature.output,
            },
        );
        hugr
    }
}
