//! Utilities for building valid HUGRs.
//!
//! This module includes various tools for building HUGRs.
//!
//! Depending on the type of HUGR you want to build, you may want to use one of
//! the following builders:
//!
//! - [`ModuleBuilder`]: For building a module with function declarations and
//!   definitions.
//! - [`DFGBuilder`]: For building a dataflow graph.
//! - [`FunctionBuilder`]: A `DFGBuilder` specialised in defining functions with a
//!   dataflow graph.
//! - [`CFGBuilder`]: For building a control flow graph.
//! - [`ConditionalBuilder`]: For building a conditional node.
//! - [`TailLoopBuilder`]: For building a tail-loop node.
//!
//! Additionally, the [`CircuitBuilder`] provides an alternative to the
//! [`DFGBuilder`] when working with circuits, where some inputs of operations directly
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
//! # use hugr::{Hugr, HugrView};
//! # use hugr::builder::{BuildError, BuildHandle, Container, DFGBuilder, Dataflow, DataflowHugr, ModuleBuilder, DataflowSubContainer, HugrBuilder};
//! use hugr::extension::prelude::bool_t;
//! use hugr::std_extensions::logic::{self, LogicOp};
//! use hugr::types::Signature;
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
//!             Signature::new_endo(bool_t()),
//!         )?;
//!
//!         // Get the wires from the function inputs.
//!         let [w] = dfg.input_wires_arr();
//!
//!         // Add an operation connected to the input wire, and get the new dangling wires.
//!         let [w] = dfg.add_dataflow_op(LogicOp::Not, [w])?.outputs_arr();
//!
//!         // Finish the function, connecting some wires to the output.
//!         dfg.finish_with_outputs([w])
//!     }?;
//!
//!     // Add a similar function, using the circuit builder interface.
//!     let _circuit_handle = {
//!         let mut dfg = module_builder.define_function(
//!             "circuit",
//!             Signature::new_endo(vec![bool_t(), bool_t()]),
//!         )?;
//!         let mut circuit = dfg.as_circuit(dfg.input_wires());
//!
//!         // Add multiple operations, indicating only the wire index.
//!         circuit.append(LogicOp::Not, [0])?.append(LogicOp::Not, [1])?;
//!
//!         // Finish the circuit, and return the dataflow graph after connecting its outputs.
//!         let outputs = circuit.finish();
//!         dfg.finish_with_outputs(outputs)
//!     }?;
//!
//!     // Finish building the HUGR, consuming the builder.
//!     //
//!     // Requires a registry with all the extensions used in the module.
//!     module_builder.finish_hugr()
//! }?;
//!
//! // The built HUGR is always valid.
//! hugr.validate().unwrap_or_else(|e| {
//!     panic!("HUGR validation failed: {e}");
//! });
//! # Ok(())
//! # }
//! # doctest().unwrap();
//! ```
use thiserror::Error;

use crate::extension::SignatureError;
use crate::extension::simple_op::OpLoadError;
use crate::hugr::ValidationError;
use crate::hugr::linking::NodeLinkingError;
use crate::ops::handle::{BasicBlockID, CfgID, ConditionalID, DfgID, FuncID, TailLoopID};
use crate::ops::{NamedOp, OpType};
use crate::types::Type;
use crate::types::{ConstTypeError, Signature, TypeRow};
use crate::{Node, Port, Wire};

pub mod handle;
pub use handle::BuildHandle;

mod build_traits;
pub use build_traits::{
    Container, Dataflow, DataflowHugr, DataflowSubContainer, HugrBuilder, SubContainer,
};

pub mod dataflow;
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

/// Return a `FunctionType` with the same input and output types (specified).
pub fn endo_sig(types: impl Into<TypeRow>) -> Signature {
    Signature::new_endo(types)
}

/// Return a `FunctionType` with the specified input and output types.
pub fn inout_sig(inputs: impl Into<TypeRow>, outputs: impl Into<TypeRow>) -> Signature {
    Signature::new(inputs, outputs)
}

#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
/// Error while building the HUGR.
pub enum BuildError {
    /// The constructed HUGR is invalid.
    #[error("The constructed HUGR is invalid: {0}.")]
    InvalidHUGR(#[from] ValidationError<Node>),
    /// `SignatureError` in trying to construct a node (differs from
    /// [`ValidationError::SignatureError`] in that we could not construct a node to report about)
    #[error(transparent)]
    SignatureError(#[from] SignatureError),
    /// Tried to add a malformed [Const]
    ///
    /// [Const]: crate::ops::constant::Const
    #[error("Constant failed typechecking: {0}")]
    BadConstant(#[from] ConstTypeError),
    /// CFG can only have one entry.
    #[error("CFG entry node already built for CFG node: {0}.")]
    EntryBuiltError(Node),
    /// We don't allow creating `BasicBlockBuilder<Hugr>`s when the sum-rows
    /// are not homogeneous. Use a `CFGBuilder` and create a valid graph instead.
    #[error(
        "Cannot initialize hugr for a BasicBlockBuilder with complex sum-rows. Use a CFGBuilder instead."
    )]
    BasicBlockTooComplex,
    /// Node was expected to have a certain type but was found to not.
    #[error("Node with index {node} does not have type {op_desc} as expected.")]
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

    /// Node not found in Hugr
    #[error("{node} not found in the Hugr")]
    NodeNotFound {
        /// Missing node
        node: Node,
    },

    /// From [Dataflow::add_link_hugr_by_node_with_wires]
    #[error{"In inserting Hugr: {0}"}]
    HugrInsertionError(#[from] NodeLinkingError<Node, Node>),

    /// From [Dataflow::add_link_view_by_node_with_wires].
    /// Note that because the type of node in the [NodeLinkingError] depends
    /// upon the view being inserted, we convert the error to a string here.
    #[error("In inserting HugrView: {0}")]
    HugrViewInsertionError(String),

    /// Wire not found in Hugr
    #[error("Wire not found in Hugr: {0}.")]
    WireNotFound(Wire),

    /// Error in `CircuitBuilder`
    #[error("Error in CircuitBuilder: {0}.")]
    CircuitError(#[from] circuit::CircuitBuildError),

    /// Invalid wires when setting outputs
    #[error("Found an error while setting the outputs of a {} container, {container_node}. {error}", .container_op.name())]
    #[allow(missing_docs)]
    OutputWiring {
        container_op: Box<OpType>,
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
        op: Box<OpType>,
        #[source]
        error: BuilderWiringError,
    },

    #[error("Failed to load an extension op: {0}")]
    #[allow(missing_docs)]
    ExtensionOp(#[from] OpLoadError),
}

#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
/// Error raised when wiring up a node during the build process.
pub enum BuilderWiringError {
    /// Tried to copy a linear type.
    #[error("Cannot copy linear type {typ} from output {src_offset} of node {src}")]
    #[allow(missing_docs)]
    NoCopyLinear {
        typ: Box<Type>,
        src: Node,
        src_offset: Port,
    },
    /// The ancestors of an inter-graph edge are not related.
    #[error(
        "Cannot connect an inter-graph edge between unrelated nodes. Tried connecting {src} ({src_offset}) with {dst} ({dst_offset})."
    )]
    #[allow(missing_docs)]
    NoRelationIntergraph {
        src: Node,
        src_offset: Port,
        dst: Node,
        dst_offset: Port,
    },
    /// Inter-Graph edges can only carry copyable data.
    #[error(
        "Inter-graph edges cannot carry non-copyable data {typ}. Tried connecting {src} ({src_offset}) with {dst} ({dst_offset})."
    )]
    #[allow(missing_docs)]
    NonCopyableIntergraph {
        src: Node,
        src_offset: Port,
        dst: Node,
        dst_offset: Port,
        typ: Box<Type>,
    },
}

#[cfg(test)]
pub(crate) mod test {
    use rstest::fixture;

    use crate::Hugr;
    use crate::extension::prelude::{bool_t, usize_t};
    use crate::hugr::{HugrMut, views::HugrView};
    use crate::ops;
    use crate::package::Package;
    use crate::types::{PolyFuncType, Signature};

    use super::handle::BuildHandle;
    use super::{
        BuildError, CFGBuilder, DFGBuilder, Dataflow, DataflowHugr, FuncID, FunctionBuilder,
        ModuleBuilder,
    };
    use super::{DataflowSubContainer, HugrBuilder};

    /// Wire up inputs of a Dataflow container to the outputs.
    pub(crate) fn n_identity<T: DataflowSubContainer>(
        dataflow_builder: T,
    ) -> Result<T::ContainerHandle, BuildError> {
        let w = dataflow_builder.input_wires();
        dataflow_builder.finish_with_outputs(w)
    }

    pub(crate) fn build_main(
        signature: PolyFuncType,
        f: impl FnOnce(FunctionBuilder<&mut Hugr>) -> Result<BuildHandle<FuncID<true>>, BuildError>,
    ) -> Result<Hugr, BuildError> {
        let mut module_builder = ModuleBuilder::new();
        let f_builder = module_builder.define_function("main", signature)?;

        f(f_builder)?;

        Ok(module_builder.finish_hugr()?)
    }

    #[fixture]
    pub(crate) fn simple_dfg_hugr() -> Hugr {
        let dfg_builder = DFGBuilder::new(Signature::new(vec![bool_t()], vec![bool_t()])).unwrap();
        let [i1] = dfg_builder.input_wires_arr();
        dfg_builder.finish_hugr_with_outputs([i1]).unwrap()
    }

    #[fixture]
    pub(crate) fn simple_funcdef_hugr() -> Hugr {
        let fn_builder =
            FunctionBuilder::new("test", Signature::new(vec![bool_t()], vec![bool_t()])).unwrap();
        let [i1] = fn_builder.input_wires_arr();
        fn_builder.finish_hugr_with_outputs([i1]).unwrap()
    }

    #[fixture]
    pub(crate) fn simple_module_hugr() -> Hugr {
        let mut builder = ModuleBuilder::new();
        let sig = Signature::new(vec![bool_t()], vec![bool_t()]);
        builder.declare("test", sig.into()).unwrap();
        builder.finish_hugr().unwrap()
    }

    #[fixture]
    pub(crate) fn simple_cfg_hugr() -> Hugr {
        let mut cfg_builder =
            CFGBuilder::new(Signature::new(vec![usize_t()], vec![usize_t()])).unwrap();
        super::cfg::test::build_basic_cfg(&mut cfg_builder).unwrap();
        cfg_builder.finish_hugr().unwrap()
    }

    #[fixture]
    pub(crate) fn simple_package() -> Package {
        let hugr = simple_module_hugr();
        Package::new([hugr])
    }

    #[fixture]
    pub(crate) fn multi_module_package() -> Package {
        let hugr0 = simple_module_hugr();
        let hugr1 = simple_module_hugr();
        Package::new([hugr0, hugr1])
    }

    /// A helper method which creates a DFG rooted hugr with Input and Output node
    /// only (no wires), given a function type with extension delta.
    // TODO consider taking two type rows and using TO_BE_INFERRED
    pub(crate) fn closed_dfg_root_hugr(signature: Signature) -> Hugr {
        let mut hugr = Hugr::new_with_entrypoint(ops::DFG {
            signature: signature.clone(),
        })
        .unwrap();
        hugr.add_node_with_parent(
            hugr.entrypoint(),
            ops::Input {
                types: signature.input,
            },
        );
        hugr.add_node_with_parent(
            hugr.entrypoint(),
            ops::Output {
                types: signature.output,
            },
        );
        hugr
    }

    /// Builds a DFG-entrypoint Hugr (no inputs, one bool_t output) containing two calls,
    /// to a FuncDefn and a FuncDecl each bool_t->bool_t.
    /// Returns the Hugr and both function handles.
    #[fixture]
    pub(crate) fn dfg_calling_defn_decl() -> (Hugr, FuncID<true>, FuncID<false>) {
        let mut dfb = DFGBuilder::new(Signature::new(vec![], bool_t())).unwrap();
        let new_defn = {
            let mut mb = dfb.module_root_builder();
            let fb = mb
                .define_function("helper_id", Signature::new_endo(bool_t()))
                .unwrap();
            let [f_inp] = fb.input_wires_arr();
            fb.finish_with_outputs([f_inp]).unwrap()
        };
        let new_decl = dfb
            .module_root_builder()
            .declare("helper2", Signature::new_endo(bool_t()).into())
            .unwrap();
        let cst = dfb.add_load_value(ops::Value::true_val());
        let [c1] = dfb
            .call(new_defn.handle(), &[], [cst])
            .unwrap()
            .outputs_arr();
        let [c2] = dfb.call(&new_decl, &[], [c1]).unwrap().outputs_arr();
        (
            dfb.finish_hugr_with_outputs([c2]).unwrap(),
            *new_defn.handle(),
            new_decl,
        )
    }
}
