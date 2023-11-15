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
//! a simple quantum extension and then use the [[DFGBuilder]] as follows:
//! ```
//! use hugr::builder::{BuildError, DFGBuilder, Dataflow, DataflowHugr};
//! use hugr::extension::prelude::{BOOL_T, QB_T};
//! use hugr::hugr::Hugr;
//! use hugr::type_row;
//! use hugr::types::FunctionType;
//!
//! mod mini_quantum_extension {
//!     use smol_str::SmolStr;
//!
//!     use hugr::{
//!         extension::{
//!             prelude::{BOOL_T, QB_T},
//!             ExtensionId, ExtensionRegistry, PRELUDE,
//!         },
//!         ops::LeafOp,
//!         type_row,
//!         types::{FunctionType, PolyFuncType},
//!         Extension,
//!     };
//!
//!     use lazy_static::lazy_static;
//!
//!     fn one_qb_func() -> PolyFuncType {
//!         FunctionType::new_linear(type_row![QB_T]).into()
//!     }
//!
//!     fn two_qb_func() -> PolyFuncType {
//!         FunctionType::new_linear(type_row![QB_T, QB_T]).into()
//!     }
//!     /// The extension identifier.
//!     pub const EXTENSION_ID: ExtensionId = ExtensionId::new_unchecked("mini.quantum");
//!     fn extension() -> Extension {
//!         let mut extension = Extension::new(EXTENSION_ID);
//!
//!         extension
//!             .add_op_type_scheme_simple(SmolStr::new_inline("H"), "Hadamard".into(), one_qb_func())
//!             .unwrap();
//!
//!         extension
//!             .add_op_type_scheme_simple(SmolStr::new_inline("CX"), "CX".into(), two_qb_func())
//!             .unwrap();
//!
//!         extension
//!             .add_op_type_scheme_simple(
//!                 SmolStr::new_inline("Measure"),
//!                 "Measure a qubit, returning the qubit and the measurement result.".into(),
//!                 FunctionType::new(type_row![QB_T], type_row![QB_T, BOOL_T]).into(),
//!             )
//!             .unwrap();
//!
//!         extension
//!     }
//!
//!     lazy_static! {
//!         /// Quantum extension definition.
//!         pub static ref EXTENSION: Extension = extension();
//!         static ref REG: ExtensionRegistry = [EXTENSION.to_owned(), PRELUDE.to_owned()].into();
//!
//!     }
//!     fn get_gate(gate_name: &str) -> LeafOp {
//!         EXTENSION
//!             .instantiate_extension_op(gate_name, [], &REG)
//!             .unwrap()
//!             .into()
//!     }
//!     pub fn h_gate() -> LeafOp {
//!         get_gate("H")
//!     }
//!
//!     pub fn cx_gate() -> LeafOp {
//!         get_gate("CX")
//!     }
//!
//!     pub fn measure() -> LeafOp {
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
//! // c:              ╩═
//! fn make_dfg_hugr() -> Result<Hugr, BuildError> {
//!     let mut dfg_builder = DFGBuilder::new(FunctionType::new(
//!         type_row![QB_T, QB_T],
//!         type_row![QB_T, QB_T, BOOL_T],
//!     ))?;
//!     let [wire0, wire1] = dfg_builder.input_wires_arr();
//!     let wire2 = dfg_builder.add_dataflow_op(h_gate(), vec![wire0])?;
//!     let wire3 = dfg_builder.add_dataflow_op(h_gate(), vec![wire1])?;
//!     let wire45 = dfg_builder.add_dataflow_op(cx_gate(), wire2.outputs().chain(wire3.outputs()))?;
//!     let wire67 = dfg_builder.add_dataflow_op(measure(), wire45.outputs().last())?;
//!     dfg_builder.finish_prelude_hugr_with_outputs(wire45.outputs().take(1).chain(wire67.outputs()))
//! }
//!
//! let h: Hugr = make_dfg_hugr().unwrap();
//! let serialized = serde_json::to_string(&h).unwrap();
//! println!("{}", serialized);
//! ```
//!
//! # Optional feature flags
//!
//! - `pyo3`: Enable Python bindings via [pyo3](https://docs.rs/pyo3).
//!

#![warn(missing_docs)]
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
pub mod values;

pub use crate::core::{
    CircuitUnit, Direction, IncomingPort, Node, NodeIndex, OutgoingPort, Port, PortIndex, Wire,
};
pub use crate::extension::Extension;
pub use crate::hugr::{Hugr, HugrView, SimpleReplacement};
