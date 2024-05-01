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
//! use hugr::builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr};
//! use hugr::extension::prelude::{BOOL_T, QB_T};
//! use hugr::hugr::Hugr;
//! use hugr::type_row;
//! use hugr::types::FunctionType;
//!
//! // The type of qubits, `QB_T` is in the prelude but, by default, no gateset
//! // is defined. This module provides Hadamard and CX gates.
//! mod mini_quantum_extension {
//!     use hugr::{
//!         extension::{
//!             prelude::{BOOL_T, QB_T},
//!             ExtensionId, ExtensionRegistry, PRELUDE,
//!         },
//!         ops::{CustomOp, OpName},
//!         type_row,
//!         types::{FunctionType, PolyFuncType},
//!         Extension,
//!     };
//!
//!     use lazy_static::lazy_static;
//!
//!     fn one_qb_func() -> PolyFuncType {
//!         FunctionType::new_endo(type_row![QB_T]).into()
//!     }
//!
//!     fn two_qb_func() -> PolyFuncType {
//!         FunctionType::new_endo(type_row![QB_T, QB_T]).into()
//!     }
//!     /// The extension identifier.
//!     pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("mini.quantum");
//!     fn extension() -> Extension {
//!         let mut extension = Extension::new(EXTENSION_ID);
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
//!                 FunctionType::new(type_row![QB_T], type_row![QB_T, BOOL_T]),
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
//!     fn get_gate(gate_name: impl Into<OpName>) -> CustomOp {
//!         EXTENSION
//!             .instantiate_extension_op(&gate_name.into(), [], &REG)
//!             .unwrap()
//!             .into()
//!     }
//!     pub fn h_gate() -> CustomOp {
//!         get_gate("H")
//!     }
//!
//!     pub fn cx_gate() -> CustomOp {
//!         get_gate("CX")
//!     }
//!
//!     pub fn measure() -> CustomOp {
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
//!     let mut dfg_builder = DFGBuilder::new(FunctionType::new(
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
//!

// Unstable check, may cause false positives.
// https://github.com/rust-lang/rust-clippy/issues/5112
#![warn(clippy::debug_assert_with_mut_call)]

pub mod algorithm;
pub mod builder;
pub mod core;
pub mod extension;
pub mod hugr;
pub mod macros;
pub mod ops;
pub mod std_extensions;
pub mod types;
mod utils;

pub use crate::core::{
    CircuitUnit, Direction, IncomingPort, Node, NodeIndex, OutgoingPort, Port, PortIndex, Wire,
};
pub use crate::extension::Extension;
pub use crate::hugr::{Hugr, HugrView, SimpleReplacement};
