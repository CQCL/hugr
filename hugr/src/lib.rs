//! Extensible, graph-based program representation with first-class support for linear types.
//!
//! The name HUGR stands for "Hierarchical Unified Graph Representation". It is designed primarily
//! as an intermediate representation and interchange format for quantum and hybrid
//! classical–quantum programs.
//!
//! Both data-flow and control-flow graphs can be represented in the HUGR. Nodes in the graph may
//! represent basic operations, or may themselves have "child" graphs, which inherit their inputs
//! and outputs. Special "non-local" edges allow data to pass directly from a node to another node
//! that is not a direct descendent (subject to causality constraints).
//!
//! The specification can be found
//! [here](https://github.com/CQCL/hugr/blob/main/specification/hugr.md).
//!
//! This crate provides a Rust implementation of HUGR and the standard extensions defined in the
//! specification.
//!
//! It includes methods for:
//!
//! - building HUGRs from basic operations;
//! - defining new extensions;
//! - serializing and deserializing HUGRs;
//! - performing local rewrites.
//!
//! # Example
//!
//! To build a HUGR for a simple quantum circuit and then serialize it to a buffer, we can define
//! a simple quantum extension and then use the [[builder::DFGBuilder]] as follows:
//! ```
//! use hugr::builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr, inout_sig};
//! use hugr::extension::prelude::{BOOL_T, QB_T};
//! use hugr::hugr::Hugr;
//! use hugr::type_row;
//! use hugr::types::FuncValueType;
//!
//! // The type of qubits, `QB_T` is in the prelude but, by default, no gateset
//! // is defined. This module provides Hadamard and CX gates.
//! mod mini_quantum_extension {
//!     use hugr::{
//!         extension::{
//!             prelude::{BOOL_T, QB_T},
//!             ExtensionId, ExtensionRegistry, PRELUDE, Version,
//!         },
//!         ops::{ExtensionOp, OpName},
//!         type_row,
//!         types::{FuncValueType, OpDefSignature},
//!         Extension,
//!     };
//!
//!     use lazy_static::lazy_static;
//!
//!     fn one_qb_func() -> OpDefSignature {
//!         FuncValueType::new_endo(type_row![QB_T]).into()
//!     }
//!
//!     fn two_qb_func() -> OpDefSignature {
//!         FuncValueType::new_endo(type_row![QB_T, QB_T]).into()
//!     }
//!     /// The extension identifier.
//!     pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("mini.quantum");
//!     pub const VERSION: Version = Version::new(0, 1, 0);
//!     fn extension() -> Extension {
//!         let mut extension = Extension::new(EXTENSION_ID, VERSION);
//!
//!         extension
//!             .add_op(OpName::new_inline("H"), "Hadamard".into(), one_qb_func())
//!             .unwrap();
//!
//!         extension
//!             .add_op(OpName::new_inline("CX"), "CX".into(), two_qb_func())
//!             .unwrap();
//!
//!         extension
//!             .add_op(
//!                 OpName::new_inline("Measure"),
//!                 "Measure a qubit, returning the qubit and the measurement result.".into(),
//!                 FuncValueType::new(type_row![QB_T], type_row![QB_T, BOOL_T]),
//!             )
//!             .unwrap();
//!
//!         extension
//!     }
//!
//!     lazy_static! {
//!         /// Quantum extension definition.
//!         pub static ref EXTENSION: Extension = extension();
//!         static ref REG: ExtensionRegistry =
//!             ExtensionRegistry::try_new([EXTENSION.to_owned(), PRELUDE.to_owned()]).unwrap();
//!
//!     }
//!     fn get_gate(gate_name: impl Into<OpName>) -> ExtensionOp {
//!         EXTENSION
//!             .instantiate_extension_op(&gate_name.into(), [], &REG)
//!             .unwrap()
//!             .into()
//!     }
//!     pub fn h_gate() -> ExtensionOp {
//!         get_gate("H")
//!     }
//!
//!     pub fn cx_gate() -> ExtensionOp {
//!         get_gate("CX")
//!     }
//!
//!     pub fn measure() -> ExtensionOp {
//!         get_gate("Measure")
//!     }
//! }
//!
//! use mini_quantum_extension::{cx_gate, h_gate, measure};
//!
//! //      ┌───┐
//! // q_0: ┤ H ├──■─────
//! //      ├───┤┌─┴─┐┌─┐
//! // q_1: ┤ H ├┤ X ├┤M├
//! //      └───┘└───┘└╥┘
//! // c:              ╚═
//! fn make_dfg_hugr() -> Result<Hugr, BuildError> {
//!     let mut dfg_builder = DFGBuilder::new(inout_sig(
//!         type_row![QB_T, QB_T],
//!         type_row![QB_T, QB_T, BOOL_T],
//!     ))?;
//!     let [wire0, wire1] = dfg_builder.input_wires_arr();
//!     let h0 = dfg_builder.add_dataflow_op(h_gate(), vec![wire0])?;
//!     let h1 = dfg_builder.add_dataflow_op(h_gate(), vec![wire1])?;
//!     let cx = dfg_builder.add_dataflow_op(cx_gate(), h0.outputs().chain(h1.outputs()))?;
//!     let measure = dfg_builder.add_dataflow_op(measure(), cx.outputs().last())?;
//!     dfg_builder.finish_prelude_hugr_with_outputs(cx.outputs().take(1).chain(measure.outputs()))
//! }
//!
//! let h: Hugr = make_dfg_hugr().unwrap();
//! let serialized = serde_json::to_string(&h).unwrap();
//! println!("{}", serialized);
//! ```

// These modules are re-exported as-is. If more control is needed, define a new module in this crate with the desired exports.
// The doc inline directive is necessary for renamed modules to appear as if they were defined in this crate.
pub use hugr_core::{builder, core, extension, ops, package, std_extensions, types, utils};
#[doc(inline)]
pub use hugr_passes as algorithms;

// Modules with hand-picked re-exports.
pub mod hugr;

// Top-level re-exports for convenience.
pub use hugr_core::core::{
    CircuitUnit, Direction, IncomingPort, Node, NodeIndex, OutgoingPort, Port, PortIndex, Wire,
};
pub use hugr_core::extension::Extension;
pub use hugr_core::hugr::{Hugr, HugrView, SimpleReplacement};

// Re-export macros.
pub use hugr_core::macros::{const_extension_ids, type_row};
