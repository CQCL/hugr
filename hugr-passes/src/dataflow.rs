#![warn(missing_docs)]
//! Dataflow analysis of Hugrs.

mod datalog;
pub use datalog::{AbstractValue, DFContext};

mod machine;
pub use machine::{Machine, TailLoopTermination};

mod partial_value;
pub use partial_value::{BaseValue, PVEnum, PartialSum, PartialValue, Sum};

#[cfg(test)]
mod test;
